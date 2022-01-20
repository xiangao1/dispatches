import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import json

################################################################################
############################ Helper functions ##################################
################################################################################


def df_col_to_dict(df, key_col_name, value_col_name):
    """
    A funtion to assemble a dictionary, who has keys 'key_col_name' and values
    'value_col_name' from a pandas dataframe.
    Arguments:
        key_col_name: the column which will be used as the key of the dict [str]
        value_col_name: the column which will be used as the value of the dict [str]
    Return
        the new dict
    """
    return dict(zip(df[key_col_name], df[value_col_name]))


def get_data_given(df, bus=None, date=None, hour=None, generator=None, fuel_type=None):

    """
    This function gets the data out of a pandas dataframe given one or more
    options, e.g. time.
    Arguments:
        df: the dataframe we are interested in
        bus: the bus ID we want [int]
        date: the date we want [str]
        hour: the hour we want [int]
        generator: the generator ID [str]
        fuel_type: generator fuel, e.g. Coal [str]
    Returns:
        df: a dataframe that has the information we specified.
    """

    # get data given bus id
    if bus is not None:
        # in the original rts-gmlc dataset there is a Bus ID col
        if "Bus ID" in df.columns:
            df = df.loc[(df["Bus ID"] == bus)]

        # in the prescient result data we have to extract bus id from other col
        # e.g. gennerator name col
        elif "Generator" in df.columns:
            # convert the type to str
            bus = str(bus)
            # find the rows that starts with the bus name
            searchrows = df["Generator"].str.startswith(bus)
            df = df.loc[searchrows, :]

    # get data given date
    if date is not None:
        df = df.loc[(df["Date"] == date)]

    # get data given hour
    if hour is not None:
        df = df.loc[(df["Hour"] == hour)]

    # get data given hour
    if generator is not None:

        # Similarly this is for Prescient result data
        if "Generator" in df.columns:
            df = df.loc[(df["Generator"] == generator)]
        # this is for rts-gmlc dataset
        elif "GEN UID" in df.columns:
            df = df.loc[(df["GEN UID"] == generator)]

    # get data given fuel
    if fuel_type is not None:
        df = df.loc[df["Fuel"] == fuel_type]

    return df


def sum_data_to_dict(df, node_edge_list, data_col):
    """
    A function to calculate the sum value at a specific bus/edge. For example,
    the total capacity at a bus.
    Arguments:
        df: the dataframe that has the results
        node_edge_list: a list of all the nodes or edges you interested in
        data_col: the data column you want (e.g. 'Dispatch')
    Returns:
        the resulted dictionary
    """

    # assemble sums into a dict
    result = {i: get_data_given(df, bus=i)[data_col].sum() for i in node_edge_list}

    return result


class NetworkData:
    def __init__(self, network_data_dir="../RTS-GMLC/RTS_Data/SourceData/"):
        """
        This class reads in RTS-GMLC dataset and makes network plots.
        Arguments:
            network_data_dir: directory containing RTS-GMLC raw data [string]
        """

        assert isinstance(
            network_data_dir, str
        ), "Arugment network_data_dir is the wrong type! It should a string."

        # read the bus and branch data
        self.bus_df = pd.read_csv(network_data_dir + "bus.csv")
        self.branch_df = pd.read_csv(network_data_dir + "branch.csv")

        # generator params (this has the capacity of each generator)
        self.gen_param_df = pd.read_csv(network_data_dir + "gen.csv")

        # a dictionary that maps bus id to bus name
        self.bus_id_to_bus_name = df_col_to_dict(self.bus_df, "Bus ID", "Bus Name")
        self.bus_name_to_bus_id = df_col_to_dict(self.bus_df, "Bus Name", "Bus ID")

        # thermal generators df
        dispatchable_fuel_types = ["Coal", "Oil", "NG", "Nuclear"]
        self.thermal_gen_param_df = self.gen_param_df.loc[
            self.gen_param_df["Fuel"].isin(dispatchable_fuel_types)
        ]

        # renewable generators df
        renewable_fuel_types = ["Hydro", "Solar", "Wind"]
        self.renewable_gen_param_df = self.gen_param_df.loc[
            self.gen_param_df["Fuel"].isin(renewable_fuel_types)
        ]

        # bus id
        self.bus_id = list(self.bus_df["Bus ID"])

        self.num_buses = len(self.bus_id)
        self.num_thermal_generators = len(self.thermal_gen_param_df)
        self.num_renewable_generators = len(self.renewable_gen_param_df)
        self.total_num_generator = (
            self.num_thermal_generators + self.num_renewable_generators
        )
        self.total_thermal_power = self.thermal_gen_param_df["PMax MW"].sum()
        self.total_renewable_power = self.renewable_gen_param_df["PMax MW"].sum()
        self.total_power = self.gen_param_df["PMax MW"].sum()

        self.bus_thermal_pmax_dict = sum_data_to_dict(
            self.thermal_gen_param_df, self.bus_id, data_col="PMax MW"
        )

        # bus renewable pmax dict
        self.bus_renewable_pmax_dict = sum_data_to_dict(
            self.renewable_gen_param_df, self.bus_id, data_col="PMax MW"
        )

        # bus total pmax dict
        self.bus_pmax_dict = sum_data_to_dict(
            self.gen_param_df, self.bus_id, data_col="PMax MW"
        )

        # line flow limit dict
        self.line_flow_limit_dict = df_col_to_dict(self.branch_df, "UID", "Cont Rating")

        # number of thermal units
        self.bus_unit_num_dict = {
            bus: len(get_data_given(self.thermal_gen_param_df, bus=bus))
            for bus in self.bus_id
        }

        self.bus_renewable_num_dict = {
            bus: len(get_data_given(self.renewable_gen_param_df, bus=bus))
            for bus in self.bus_id
        }

        # construct the network
        self.G = self.construct_network()

    def print_summary(self):

        """
        This function prints the summary information on RTS-GMLC dataset, e.g.
        the total number of buses.
        Arguments:
            self: class instance itself.
        Return:
            None
        """

        print("The number of buses is ", self.num_buses)
        print("The number of thermal generators is ", self.num_thermal_generators)
        print("The number of renewable generators is ", self.num_renewable_generators)
        print("The total number of generators is ", self.total_num_generator)
        print("The amount of thermal power is {} MW".format(self.total_thermal_power))
        print(
            "The amount of renewable power is {} MW".format(self.total_renewable_power)
        )
        print("The total amount of power is {} MW".format(self.total_power))

        return

    ################################################################################
    ##################### Functions to Construct the Newtork #######################
    ################################################################################
    @staticmethod
    def assemble_node_tuples(node_id, node_name, node_latitudes, node_longitudes):
        """
        This function assembles a list of tuples (node_id, dictionary of node
        properties). This list of tuple can the be passed to networkx to construct
        the network.
        Arguments:
            node_id: the id of the node, e,g. 101 [int]
            node_name: the nickname of the node, e.g. Adam [str]
            node_latitudes: the latitude of the node
            node_longitudes: the longitude of the node
        Returns:
            node_list: a list of tuples (node_id, dictionary of node properties)
        """

        num_nodes = len(node_id)
        node_list = [
            (
                node_id[i],
                {"pos": (node_latitudes[i], node_longitudes[i]), "name": node_name[i]},
            )
            for i in range(num_nodes)
        ]

        return node_list

    @staticmethod
    def assemble_edge_tuples(start_node, end_node, edge_name):
        """
        This function can assemble a list of tuples (edge start point, edge end
         point, dctionary of edge properties). This list of tuple can the
         be passed to networkx to construct the network.
        Arguments:
            start_node: the starting node id of an edge [int]
            end_node: the ending node id of an edge [int]
            edge_name: the name of the edge, e.g. A1 [str]
        Returns:
            edg_list: a list of tuples (edge start point, edge end point,
            dctionary of edge properties)
        """

        num_edges = len(edge_name)
        edge_list = [
            (start_node[i], end_node[i], {"name": edge_name[i]})
            for i in range(num_edges)
        ]

        return edge_list

    def construct_network(self):
        """
        The actual function to construct the network in networkx.
        Arguments:
            self: the class instance itself
        Returns:
            G: a networkx object which already has the RTS-GMLC network structure
        """

        # extract necessary data
        bus_id = self.bus_df["Bus ID"]
        bus_lat = self.bus_df["lat"]
        bus_lng = self.bus_df["lng"]
        bus_name = self.bus_df["Bus Name"]

        start_node = self.branch_df["From Bus"]
        end_node = self.branch_df["To Bus"]
        edge_name = self.branch_df["UID"]

        # create a network
        G = nx.Graph()

        # add nodes with coordinates as the atrributes
        nodeList = self.assemble_node_tuples(bus_id, bus_name, bus_lat, bus_lng)
        G.add_nodes_from(nodeList)

        # add edges
        edgList = self.assemble_edge_tuples(start_node, end_node, edge_name)
        G.add_edges_from(edgList)

        return G

    ################################################################################
    ##################### Functions to Plot the Newtork ############################
    ################################################################################

    # function to plot the network
    @staticmethod
    def visualize_network(
        G,
        node_info,
        node_text,
        edge_text,
        color_coded=True,
        title="RTS-GMLC Network Visualization",
        legend_label="Color Bar Label",
    ):

        """
        This function takes a network object created by networkx package
        and plot the topolgy of it. This plot is interactive: when pointed
        at each node in the network user specified info can be displayed.
        Also, the color code of the node shows the most important info of
        the plot (e.g. LMPs).
        Arguments:
            G: a network (graph) object created by networkx package
            node_info: the information user wants to be displayed as the color coding. [list]
            node_text: the text user wants to show when a node is pointed at. [list]
            edge_text: the text user wants to show when an edge is pointed at. [list]
            color_coded: whether we want to display node_info as color coded dots. [bool]
            title: figure title
            legend_label: color bar label
        Output:
            fig: a figure object of plotly.
        """

        ########## Extract Node and Edges Position ##########
        edge_x = []
        edge_y = []
        for edge in G.edges():

            # extract the edge coordinates
            x0, y0 = G.nodes[edge[0]]["pos"]
            x1, y1 = G.nodes[edge[1]]["pos"]

            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        # scatter the edges
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        for node in G.nodes():

            # extract the edge coordinates
            x, y = G.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        # scatter the nodes
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=color_coded,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                # colorscale='YlGnBu',
                colorscale="YlOrRd",
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15, title=legend_label, xanchor="left", titleside="right"
                ),
                line_width=2,
            ),
        )

        # this is for edge info display
        # add nodes at the middle point of the actual nodes
        edgeMid_x = []
        edgeMid_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]["pos"]
            x1, y1 = G.nodes[edge[1]]["pos"]
            edgeMid_x.append((x0 + x1) / 2)
            edgeMid_y.append((y0 + y1) / 2)

        # scatter the invisible nodes
        etexts_trace = go.Scatter(
            x=edgeMid_x, y=edgeMid_y, mode="markers", hoverinfo="text", marker_size=0.5
        )

        ########## End Extract Node and Edges Position ##########

        # set the node text and info
        if color_coded:
            node_trace.marker.color = node_info
        node_trace.text = node_text
        etexts_trace.text = edge_text

        # make the actual figure
        fig = go.Figure(
            data=[edge_trace, node_trace, etexts_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # uncomment the line below if you want white background
        fig.update_layout(template="plotly_white")

        return fig

    ################################################################################
    ##################### Other Plotting Functions #################################
    ################################################################################

    def plot_network(self, title=None, legend_label=None, color_coded=False):

        """
        Just visualize the network and does not dispaly more information.
        Arguments:
            self: the class instance itself
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded dots. [bool]
        returns:
            fig: the plotly figure object
        """

        fig = self.visualize_network(
            self.G,
            node_info=[],
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig

    def plot_thermal_generator_number(
        self,
        title="Number of Thermal Generators in RTS-GMLC",
        legend_label="Number of Thermal Generators",
        color_coded=True,
    ):

        """
        Plot the number of thermal generator at each node in a network plot.
        Arguments:
            self: the class instance itself
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        returns:
            fig: the plotly figure object
        """

        node_info = [self.bus_unit_num_dict[bus] for bus in self.G.nodes()]

        fig = self.visualize_network(
            self.G,
            node_info=node_info,
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig

    def plot_renewable_generator_number(
        self,
        title="Number of Renewable Generators in RTS-GMLC",
        legend_label="Number of Renewable Generators",
        color_coded=True,
    ):

        """
        Plot the number of renewable generator at each node in a network plot.
        Arguments:
            self: the class instance itself
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        returns:
            fig: the plotly figure object
        """

        node_info = [self.bus_renewable_num_dict[bus] for bus in self.G.nodes()]

        fig = self.visualize_network(
            self.G,
            node_info=node_info,
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig

    def plot_thermal_capacity(
        self,
        title="Thermal Capacity in RTS-GMLC",
        legend_label="Thermal Capacity [MW]",
        color_coded=True,
    ):

        """
        Plot the total capacity of thermal generator at each node in a network plot.
        Arguments:
            self: the class instance itself
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        returns:
            fig: the plotly figure object
        """

        node_info = [self.bus_thermal_pmax_dict[bus] for bus in self.G.nodes()]

        fig = self.visualize_network(
            self.G,
            node_info=node_info,
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig

    def plot_renewable_capacity(
        self,
        title="Renewable Capacity in RTS-GMLC",
        legend_label="Renewable Capacity [MW]",
        color_coded=True,
    ):

        """
        Plot the total capacity of renewable generator at each node in a network plot.
        Arguments:
            self: the class instance itself
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded
            dots.[bool]
        returns:
            fig: the plotly figure object
        """

        node_info = [self.bus_renewable_pmax_dict[bus] for bus in self.G.nodes()]

        fig = self.visualize_network(
            self.G,
            node_info=node_info,
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig

    def plot_capacity_given_fuel_type(
        self, fuel, title=None, legend_label=None, color_coded=True
    ):

        """
        Plot the total capacity of generators who use the specified fuel at each
         node in a network plot.
        Arguments:
            self: the class instance itself
            fuel: the fuel generator uses, e.g. Coal [str]
            title: title of the figure [str]
            legend_label: legend label of the color bar [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        returns:
            fig: the plotly figure object
        """

        # default title and legend label
        if title is None:
            title = "{} Generator Capacity in RTS-GMLC".format(fuel)
        if legend_label is None:
            legend_label = "{} Generator Capacity [MW]".format(fuel)

        gen_param_fuel_df = get_data_given(self.gen_param_df, fuel_type=fuel)

        bus_gen_capacity_fuel_dict = sum_data_to_dict(
            gen_param_fuel_df, self.bus_id, data_col="PMax MW"
        )

        node_info = [bus_gen_capacity_fuel_dict[bus] for bus in self.G.nodes()]

        fig = self.visualize_network(
            self.G,
            node_info=node_info,
            node_text=[],
            edge_text=[],
            title=title,
            legend_label=legend_label,
            color_coded=color_coded,
        )
        # fig.show()

        return fig


class PrescientSimulationData(NetworkData):

    """Add methods to this class to visualize Prescient results for a single
    generator. These visualizations will NOT depend on the network structure,
    but generator parameters, such as Pmax are pulled from RTS-GMLC dataset.
    """

    HighPrice = 100

    def __init__(
        self,
        result_data_dir,
        network_data_dir="../RTS-GMLC/RTS_Data/SourceData/",
        custom_string="bidding=False_",
        custom_string2="track_sced_",
    ):
        """
        This class reads in Prescient simulation results, e.g. thermal_detail.csv,
        and using the data to make plots. This class is a subclass of NetworkData,
        because some data, e.g. the nameplate capacity, is only available in
        that dataset.
        Arguments:
            result_data_dir: directory containing results from Prescient simulation [string]
            network_data_dir: directory containing RTS-GMLC raw data [string]
            custom_string: string added to middle of Prescient results file names [string]
        """

        # inherit from network data class
        super().__init__(network_data_dir)

        assert isinstance(
            result_data_dir, str
        ), "Arugment result_data_dir is the wrong type! It should be a string."
        assert isinstance(
            custom_string, str
        ), "Arugment custom_string is the wrong type! It should be a string."

        self.result_data_dir = result_data_dir
        self.custom_string = custom_string
        self.custom_string2 = custom_string2

        self.read_result_files()

        self.bus_detail_df = self.bus_detail_df.merge(
            self.bus_df[["Bus Name", "Bus ID"]],
            how="left",
            left_on="Bus",
            right_on="Bus Name",
        ).drop(columns=["Bus Name"])

        self.thermal_detail_df["Bus ID"] = (
            self.thermal_detail_df["Generator"]
            .str.split(pat="_", expand=True)[0]
            .astype(int)
        )

    def read_result_files(self):
        """
        This function reads full Prescient result files and set them as class
        attributes.
        """

        # bus details (this has LMP)
        self.bus_detail_df = pd.read_csv(
            self.result_data_dir + self.custom_string + "bus_detail.csv"
        )

        # thermal detail (this has the power delivered from each generator)
        self.thermal_detail_df = pd.read_csv(
            self.result_data_dir
            + self.custom_string
            + self.custom_string2
            + "thermal_detail.csv"
        )

        # renewable details
        self.renewable_detail_df = pd.read_csv(
            self.result_data_dir + "renewables_detail.csv"
        )

        # line detail (this has the power flow on each line)
        self.line_detail_df = pd.read_csv(self.result_data_dir + "line_detail.csv")

        # hourly summary
        self.hourly_summary_df = pd.read_csv(
            self.result_data_dir + "hourly_summary.csv"
        )

        return

    def summarize_results(self, result_num_decimals=4, include_generator_param=False):
        """
        This function summarizes the results for each generator into a dataframe.
        The summary information includes:
            Energy Delivered: MWh
            Energy Averaged Price: $/MWh
            Time On: Hours
            Time On: Fraction
            Capacity Factor: MWh (delivered) / MWh (if at 100% all time)
            Startup: Number of start-up events
            Shutdown: Number of shhut-down events
            Average Time On: hours
            Average Time Off: hours
            Total Uplift Payments: $
            Number of Uplift Days
            Scaled Mileage: sum |Power(t=i+1) - Power(t=i) | / Name Plate
            Generator Characteristics
        Arguments:
            include_generator_param: if True, add generator characteristics from
            RTS-GMLC dataset to the summary dataframe [bool]
        Returns:
            df: the summary dataframe
        """

        df = pd.DataFrame()

        # number of hours
        df["Number of Hours"] = self.thermal_detail_df.groupby("Generator")[
            "Hour"
        ].count()

        # pmax
        thermal_gen_param_df = self.thermal_gen_param_df.set_index("GEN UID")
        df["Pmax [MW]"] = thermal_gen_param_df["PMax MW"]

        # sum summary total
        sum_cols = [
            "Dispatch",
            "Dispatch DA",
            "Unit State",
            "Unit Cost",
            "Unit Uplift Payment",
            "Unit Market Revenue",
        ]
        summary_sum_cols = [
            "Total Dispatch [MW]",
            "Total DA Dispatch [MW]",
            "Total Online Time [hr]",
            "Total Cost [$]",
            "Total Uplift Payment [$]",
            "Total Revenue [$]",
        ]
        df[summary_sum_cols] = self.thermal_detail_df.groupby("Generator")[
            sum_cols
        ].sum()
        df["Total Offline Time [hr]"] = (
            df["Number of Hours"] - df["Total Online Time [hr]"]
        )
        df["Total Profit [$]"] = (
            df["Total Revenue [$]"]
            + df["Total Uplift Payment [$]"]
            - df["Total Cost [$]"]
        )

        # calculate mileage
        self.thermal_detail_df["temp"] = (
            self.thermal_detail_df.groupby(by="Generator")["Dispatch"]
            .diff()
            .fillna(0)
            .abs()
        )
        df["Abs Mileage [MW]"] = self.thermal_detail_df.groupby(["Generator"])[
            "temp"
        ].sum()
        self.thermal_detail_df.drop(columns=["temp"], inplace=True)

        # calculate shut down and start ups
        self.thermal_detail_df["Unit State"] = self.thermal_detail_df[
            "Unit State"
        ].astype(int)
        self.thermal_detail_df["temp"] = (
            self.thermal_detail_df.groupby(by="Generator")["Unit State"]
            .diff()
            .fillna(0)
        )
        df["Total Start-up"] = self.thermal_detail_df.groupby("Generator")[
            "temp"
        ].value_counts()[:, 1]
        df["Total Shut-down"] = self.thermal_detail_df.groupby("Generator")[
            "temp"
        ].value_counts()[:, -1]
        self.thermal_detail_df.drop(columns=["temp"], inplace=True)

        # calculate the average online and offline time
        def calc_mean_online_hours(df):
            unit_state = list(df["Unit State"])
            result = self.calc_on_off_hours(unit_state)
            return sum(result[1]) / len(result[1])

        def calc_mean_offline_hours(df):
            unit_state = list(df["Unit State"])
            result = self.calc_on_off_hours(unit_state)
            return sum(result[0]) / len(result[0])

        df["Average Online Time [hr]"] = self.thermal_detail_df.groupby(
            "Generator"
        ).apply(calc_mean_online_hours)
        df["Average Offline Time [hr]"] = self.thermal_detail_df.groupby(
            "Generator"
        ).apply(calc_mean_offline_hours)

        # fraction
        fraction_cols = [
            "Total Online Time [hr]",
            "Total Offline Time [hr]",
            "Total Start-up",
            "Total Shut-down",
        ]
        summary_fraction_cols = [
            "Online Fraction [%]",
            "Offline Fraction [%]",
            "Start-up Fraction [%]",
            "Shut-down Fraction [%]",
        ]
        df[summary_fraction_cols] = (
            df[fraction_cols].div(df["Number of Hours"], axis=0) * 100
        )

        # pmax fraction
        pmax_frac_cols = ["Total Dispatch [MW]", "Abs Mileage [MW]"]
        summary_pmax_frac_cols = ["Capacity Factor [%]", "Scaled Total Mileage [%]"]
        df[summary_pmax_frac_cols] = (
            df[pmax_frac_cols].div((df["Number of Hours"] * df["Pmax [MW]"]), axis=0)
            * 100
        )

        df["Total Uplift Payment Days"] = self.thermal_detail_df.groupby(
            "Generator"
        ).apply(lambda df: len(df.loc[df["Unit Uplift Payment"] > 0]))

        # LMP related cols
        self.thermal_detail_df = self.thermal_detail_df.merge(
            self.bus_detail_df[["Bus ID", "Date", "Hour", "LMP", "LMP DA"]],
            how="left",
            left_on=["Bus ID", "Date", "Hour"],
            right_on=["Bus ID", "Date", "Hour"],
        )
        avg_cols = ["LMP", "LMP DA"]
        summary_avg_cols = ["Average LMP [$/MWh]", "Average DA LMP [$/MWh]"]

        df[summary_avg_cols] = self.thermal_detail_df.groupby("Generator")[
            avg_cols
        ].mean()

        df["Number of Hours with High Price"] = self.thermal_detail_df.groupby(
            "Generator"
        ).apply(lambda df: len(df.loc[df["LMP"] >= self.HighPrice]))

        # include gen params
        if include_generator_param:
            df = df.merge(
                right=self.thermal_gen_param_df,
                how="left",
                left_on="Generator",
                right_on="GEN UID",
            )

        return df

    @staticmethod
    def calc_on_off_hours(unit_state):

        """
        This function calculates the lengths of online and offline periods given
        a list of unit states.
        Arguments:
            unit_state: a list/array of generator on/off state. 1 is on and 0 is
            off.
        Return:
            result: a dictionary whose key are 0 and 1, the values are list of
            online and offline hours.
        Example:
            Given a list of states: unit_state = [1,1,1,0,0,0,1,1].
            It will return:
            result = {0: [3], 1: [3,2]
        """

        result = {0: [], 1: []}

        fast = 1
        slow = 0

        n = len(unit_state)

        while fast < n:

            if unit_state[fast] != unit_state[slow]:
                result[unit_state[slow]].append(fast - slow)
                slow = fast
            fast += 1

        result[unit_state[slow]].append(n - slow)

        for v in result.values():
            if len(v) == 0:
                v.append(0)

        return result

    ################################################################################
    ############################ Other Plotting functions ##########################
    ################################################################################

    def price_histogram_given_time(self, date, hour):

        """
        plot the histogram of lmps given date and hour across all the nodes.
        Arguments:
            date: the date we are interested in. [str]
            hour: the hour we are interested in. [int]
        """

        bus_time_detail_df = get_data_given(self.bus_detail_df, date=date, hour=hour)
        lmp = bus_time_detail_df["LMP"]
        ax = lmp.hist(bins=10)

        ax.set_xlabel("LMP [$/MWh]", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_title("{} Hour = {} LMP Histogram".format(date, hour), fontsize=15)

        return ax


class Visualizer:

    """
    Use case 1: visualize prescient simulation results at a single hour
    Use case 2: visualize Prescient simulation results at two hours (difference)
    Use case 3: visualize difference between two Prescient simulation results at the same hour
    """

    def __init__(self, network_data, sim_data1, date1, hour1):
        """
        This class combines the network data and Prescient simulation results to
        make plots.
        We agreed that one network + one Prescient result per Visualization instance.
        We can write some static methods that operate on 2 Visualization instances.
        Arguments:
            network_data: instance of PrescientSimulationData
            sim_data1: instance of PrescientSimulationData
            date1: simulation date we are interested [str]
            hour1: simulation hour we are interested [int]
        """

        assert isinstance(
            network_data, NetworkData
        ), "Arugment network_data is the wrong type! It should be an instance of NetworkData."
        assert isinstance(
            sim_data1, PrescientSimulationData
        ), "Arugment sim_data1 is the wrong type! It should be an instance of PrescientSimulationData."

        self.network = network_data
        self.sim1 = sim_data1

        self.date1 = date1
        self.hour1 = hour1

        self.bus_id = self.network.bus_id

        # Prepare the dictionaries for plotting, save inside this class.

        # bus thermal pmax dict
        self.bus_thermal_pmax_dict = self.network.bus_thermal_pmax_dict

        # bus renewable pmax dict
        self.bus_renewable_pmax_dict = self.network.bus_renewable_pmax_dict

        # bus total pmax dict
        self.bus_pmax_dict = self.network.bus_pmax_dict

        # line flow limit dict
        self.line_flow_limit_dict = self.network.line_flow_limit_dict

        # number of thermal units
        self.bus_unit_num_dict = self.network.bus_unit_num_dict

        self.bus_id_to_bus_name = self.network.bus_id_to_bus_name
        self.bus_name_to_bus_id = self.network.bus_name_to_bus_id

        # prepare the temporal result dicts
        (
            self.lmp_dict,
            self.bus_power_delivered_dict,
            self.line_flow_dict,
            self.bus_renewable_delivered_dict,
            self.tot_generation_dict,
            self.committed_dict,
            self.tot_demand_dict,
            self.bus_generator_dispatch_dict,
            self.bus_renewable_generator_output_dict,
            self.online_hour_dict,
            self.offline_hour_dict,
        ) = self.prepare_data_dicts()

    def _plot_driver_iterate_nodes(
        self,
        data_dict,
        scale_data_dict=None,
        scaled=True,
        title=None,
        legend=None,
        color_coded=True,
        detailed_bus_info=False,
    ):

        """
        This function has the common routine to visualize results on a network
        plot.
        Arguments:
            data_dict: main data we want to plot, e.g. dispatch. [dict]
            scale_data_dict: data we use to scale the main data, e.g. pmax. [dict]
            scaled: whether scale or not. [bool]
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # construct the network
        G = self.network.G

        if scale_data_dict is None:
            result_list = [float(data_dict[bus]) for bus in G.nodes()]
        else:
            result_list = self.assemble_percent_list(
                G, data_dict, scale_data_dict, scale=scaled
            )

        # assemble the displaying texts
        node_text, line_text = self.assemble_displaying_texts(
            G, detailed_bus_info=detailed_bus_info
        )

        fig = self.network.visualize_network(
            G,
            result_list,
            node_text,
            line_text,
            color_coded=color_coded,
            title=title,
            legend_label=legend,
        )
        # fig.show()

        return fig

    def dispatched_power(self, scaled=False, title=None, legend=None, color_coded=True):

        """
        Plot the total dispatch at each bus on a network plot.
        Arguments:
            scaled: scale the dispatch by Pmax or not. [bool]
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # set default legend and title
        if title is None:
            if scaled:
                title = "{} Hour {} Scaled Power Dispatch in RTS-GMLC".format(
                    self.date1, self.hour1
                )
            else:
                title = "{} Hour {} Power Dispatch in RTS-GMLC".format(
                    self.date1, self.hour1
                )

        if legend is None:
            if scaled:
                legend = "Scaled Power Dispatch"
            else:
                legend = "Power Dispatch [MW]"

        fig = self._plot_driver_iterate_nodes(
            self.bus_power_delivered_dict,
            scale_data_dict=self.bus_thermal_pmax_dict,
            scaled=scaled,
            title=title,
            legend=legend,
            color_coded=color_coded,
            detailed_bus_info=True,
        )

        return fig

    def renewable_power(self, scaled=False, title=None, legend=None, color_coded=True):

        """
        Plot the total renewable output at each bus on a network plot.
        Arguments:
            scaled: scale the output by Pmax or not. [bool]
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # set default legend and title
        if title is None:
            if scaled:
                title = "{} Hour {} Scaled Renewable Power Output in RTS-GMLC".format(
                    self.date1, self.hour1
                )
            else:
                title = "{} Hour {} Renewable Power Output in RTS-GMLC".format(
                    self.date1, self.hour1
                )

        if legend is None:
            if scaled:
                legend = "Scaled Renewable Power Output"
            else:
                legend = "Renewable Power Output [MW]"

        fig = self._plot_driver_iterate_nodes(
            self.bus_renewable_delivered_dict,
            scale_data_dict=self.bus_renewable_pmax_dict,
            scaled=scaled,
            title=title,
            legend=legend,
            color_coded=color_coded,
        )

        return fig

    def demand(self, title=None, legend=None, color_coded=True):

        """
        Plot the demand at each bus on a network plot.
        Arguments:
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # set default legend and title
        if title is None:
            title = "{} Hour {} Demand in RTS-GMLC".format(self.date1, self.hour1)

        if legend is None:
            legend = "Demand [MW]"

        fig = self._plot_driver_iterate_nodes(
            self.tot_demand_dict,
            scale_data_dict=None,
            scaled=False,
            title=title,
            legend=legend,
            color_coded=color_coded,
        )

        return fig

    def units_committed(self, scaled=False, title=None, legend=None, color_coded=True):

        """
        Plot the number of committed generators at a bus on a network plot.
        Arguments:
            scaled: scale the number by total thermal generator at that bus or not. [bool]
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # set default legend and title
        if title is None:
            if scaled:
                title = "{} Hour {} Scaled Number of Committed Generators in RTS-GMLC".format(
                    self.date1, self.hour1
                )
            else:
                title = "{} Hour {} Number of Committed Generators in RTS-GMLC".format(
                    self.date1, self.hour1
                )

        if legend is None:
            if scaled:
                legend = "Scaled Number of Committed Generators"
            else:
                legend = "Number of Committed Generators"

        fig = self._plot_driver_iterate_nodes(
            self.committed_dict,
            scale_data_dict=self.bus_unit_num_dict,
            scaled=scaled,
            title=title,
            legend=legend,
            color_coded=color_coded,
        )

        return fig

    def prices(self, title=None, legend=None, color_coded=True):

        """
        Plot the lmp at each bus on a network plot.
        Arguments:
            title: plot title [str]
            legend: color bar legend title [str]
            color_coded: whether we want to display node_info as color coded
            dots. [bool]
        Returns:
            fig: a plotly figure object
        """

        # set default legend and title
        if title is None:
            title = "{} Hour {} Locational Marginal Prices in RTS-GMLC".format(
                self.date1, self.hour1
            )

        if legend is None:
            legend = "Locational Marginal Prices [$/MWh]"

        fig = self._plot_driver_iterate_nodes(
            self.lmp_dict,
            scale_data_dict=None,
            scaled=False,
            title=title,
            legend=legend,
            color_coded=color_coded,
        )

        return fig

    ################################################################################
    ############################ Helper functions ##################################
    ################################################################################

    def prepare_data_dicts(self):

        """
        A function to assemble dictionaries who have bus id/edge id as keys and
        the information we want to show on the network plot as the values. The
        reason for using dicts is to make sure there is no mistake when transfering
        these info to plotly.
        Arguments:
        1. date: the date we are interested in
        2. hour: the hour we are interested in
        """

        sim = self.sim1
        date = self.date1
        hour = self.hour1

        # lmp
        bus_time_detail_df = get_data_given(sim.bus_detail_df, date=date, hour=hour)
        bus_time_detail_df.insert(
            0,
            "Bus ID",
            [self.bus_name_to_bus_id[name] for name in bus_time_detail_df["Bus"]],
            True,
        )
        lmp_dict = df_col_to_dict(bus_time_detail_df, "Bus ID", "LMP")

        # bus total power delivered
        thermal_time_detail_df = get_data_given(
            sim.thermal_detail_df, date=date, hour=hour
        )
        bus_power_delivered_dict = sum_data_to_dict(
            thermal_time_detail_df, self.bus_id, data_col="Dispatch"
        )

        # line flow
        line_time_detail_df = get_data_given(sim.line_detail_df, date=date, hour=hour)
        line_flow_dict = df_col_to_dict(line_time_detail_df, "Line", "Flow")

        # renewable generation
        renewable_time_detail_df = get_data_given(
            sim.renewable_detail_df, date=date, hour=hour
        )
        bus_renewable_delivered_dict = sum_data_to_dict(
            renewable_time_detail_df, self.bus_id, data_col="Output"
        )

        # number of unit committed
        committed_time_df = get_data_given(sim.thermal_detail_df, date=date, hour=hour)
        committed_dict = sum_data_to_dict(
            committed_time_df, self.bus_id, data_col="Unit State"
        )

        # total generation
        tot_generation_dict = {
            bus: bus_renewable_delivered_dict[bus] + bus_power_delivered_dict[bus]
            for bus in self.bus_id
        }

        # total demand (this formulation is not correct)
        tot_demand_dict = {
            bus: tot_generation_dict[bus]
            - bus_time_detail_df.loc[bus_time_detail_df["Bus ID"] == bus][
                "Overgeneration"
            ]
            + bus_time_detail_df.loc[bus_time_detail_df["Bus ID"] == bus]["Shortfall"]
            for bus in self.bus_id
        }

        # dispatch per generator
        bus_generator_dispatch_dict = {
            bus: df_col_to_dict(
                get_data_given(sim.thermal_detail_df, date=date, hour=hour, bus=bus),
                "Generator",
                "Dispatch",
            )
            for bus in self.bus_id
        }

        bus_renewable_generator_output_dict = {
            bus: df_col_to_dict(
                get_data_given(sim.renewable_detail_df, date=date, hour=hour, bus=bus),
                "Generator",
                "Output",
            )
            for bus in self.bus_id
        }

        # online and offline time
        online_hour_dict, offline_hour_dict = self.find_online_and_offline_hours()

        return (
            lmp_dict,
            bus_power_delivered_dict,
            line_flow_dict,
            bus_renewable_delivered_dict,
            tot_generation_dict,
            committed_dict,
            tot_demand_dict,
            bus_generator_dispatch_dict,
            bus_renewable_generator_output_dict,
            online_hour_dict,
            offline_hour_dict,
        )

    def find_online_and_offline_hours(self):

        """
        This function can calculate the hours generators have been online and
        offline and return dicts of dicts. The outer dicts have keys of bus id.
        The inner dicts have keys of generator id in that bus and the values are
        the online hours and offline hours.
        Arugments:
            self
        Returns:
            online_hour_dict: dict that stores how long has a generator been on
            offline_hour_dict: dict that stores how long has a generator been off
        """

        online_hour_dict = {}
        offline_hour_dict = {}

        # loop thru the bus and generators
        for bus in self.bus_id:

            online_hour_dict[bus] = {}
            offline_hour_dict[bus] = {}

            # get the generators at this bus
            generator_list = list(
                get_data_given(self.network.thermal_gen_param_df, bus=bus)["GEN UID"]
            )

            for generator in generator_list:

                online_hour_dict[bus][generator] = 0
                offline_hour_dict[bus][generator] = 0

                # the last idx corresponding to the hour and date
                idx = get_data_given(
                    self.sim1.thermal_detail_df,
                    date=self.date1,
                    hour=self.hour1,
                    bus=bus,
                    generator=generator,
                ).index[0]

                state_list = list(self.sim1.thermal_detail_df.loc[:idx, "Unit State"])

                # calculate how many hours
                num_hours = 1
                while state_list[-num_hours - 1] == state_list[-1]:
                    num_hours += 1

                # online or offline?
                if state_list[-1] > 0:
                    online_hour_dict[bus][generator] = num_hours
                else:
                    offline_hour_dict[bus][generator] = num_hours

        return online_hour_dict, offline_hour_dict

    # function to assemble displaying text
    def assemble_displaying_texts(self, G, detailed_bus_info=False):

        """
        Assemble 2 lists: node_text and line_text. These 2 list contains strings
        of information that we want to display on the network plot interactively.
        Arguments:
            G: the networkx object that has the rts-gmlc network structure
        Returns:
            node_text: a list of strings that we are going to display at each
            node when being pointed at
            line_text: a list of strings that we are going to display at each
            edge when being pointed at
        """

        # assemble the percent lists
        thermal_power_percent_list = self.assemble_percent_list(
            G, self.bus_power_delivered_dict, self.bus_thermal_pmax_dict
        )
        renewable_power_percent_list = self.assemble_percent_list(
            G, self.bus_renewable_delivered_dict, self.bus_renewable_pmax_dict
        )
        tot_power_percent_list = self.assemble_percent_list(
            G, self.tot_generation_dict, self.bus_pmax_dict
        )

        node_text = []

        # loop thru the bus/node in the graph
        for idx, g in enumerate(G.nodes()):

            # g is the id of the bus #

            if not detailed_bus_info:
                text = (
                    "Bus {0} <br>".format(g)
                    + "LMP = {0:.2f}$/MWh <br>".format(self.lmp_dict[g])
                    + "Dispatchable Power = {0:.2f}MW ({1:.2f}%)<br>".format(
                        self.bus_power_delivered_dict[g],
                        thermal_power_percent_list[idx] * 100,
                    )
                    + "Renewable Power = {0:.2f}MW ({1:.2f}%)<br>".format(
                        self.bus_renewable_delivered_dict[g],
                        renewable_power_percent_list[idx] * 100,
                    )
                    + "Total Power = {0:.2f}MW ({1:.2f}%)".format(
                        self.tot_generation_dict[g], tot_power_percent_list[idx] * 100
                    )
                )

            else:

                # basic text
                text_basic = (
                    "Bus {0} <br>".format(g)
                    + "LMP = {0:.2f}$/MWh <br>".format(self.lmp_dict[g])
                    + "Dispatchable Power = {0:.2f}MW ({1:.2f}%)<br>".format(
                        self.bus_power_delivered_dict[g],
                        thermal_power_percent_list[idx] * 100,
                    )
                    + "Renewable Power = {0:.2f}MW ({1:.2f}%)<br>".format(
                        self.bus_renewable_delivered_dict[g],
                        renewable_power_percent_list[idx] * 100,
                    )
                    + "Total Power = {0:.2f}MW ({1:.2f}%)<br>".format(
                        self.tot_generation_dict[g], tot_power_percent_list[idx] * 100
                    )
                )

                # dispatchable generator text
                text_gen = ""
                for gen in self.bus_generator_dispatch_dict[g]:

                    text_gen += "{0} : {1:.2f} MW ({2:.2f}%) ".format(
                        gen,
                        self.bus_generator_dispatch_dict[g][gen],
                        self.bus_generator_dispatch_dict[g][gen]
                        / float(
                            get_data_given(self.network.gen_param_df, generator=gen)[
                                "PMax MW"
                            ]
                        )
                        * 100,
                    )

                    if self.online_hour_dict[g][gen] > 0:
                        state_hour_text = "Online {0} Hours<br>".format(
                            self.online_hour_dict[g][gen]
                        )
                    elif self.offline_hour_dict[g][gen] > 0:
                        state_hour_text = "Offline {0} Hours<br>".format(
                            self.offline_hour_dict[g][gen]
                        )

                    text_gen = text_gen + state_hour_text

                # renewable generator text
                text_renew = ""
                for gen in self.bus_renewable_generator_output_dict[g]:

                    text_gen += "{0} : {1:.2f} MW ({2:.2f}%)<br>".format(
                        gen,
                        self.bus_renewable_generator_output_dict[g][gen],
                        self.bus_renewable_generator_output_dict[g][gen]
                        / float(
                            get_data_given(self.network.gen_param_df, generator=gen)[
                                "PMax MW"
                            ]
                        )
                        * 100,
                    )

                text = text_basic + text_gen + text_renew

            node_text.append(text)

        line_text = []
        # loop thru the lines/edges in the graph
        for e in G.edges():

            # extract the line name
            line_name = G.edges[e]["name"]

            text = "Line {0} <br>".format(
                line_name
            ) + "Flow = {0:.2f}MW ({1:.2f}%)<br>".format(
                self.line_flow_dict[line_name],
                self.line_flow_dict[line_name]
                / self.line_flow_limit_dict[line_name]
                * 100,
            )

            line_text.append(text)

        return node_text, line_text

    @staticmethod
    def assemble_percent_list(G, bus_time_dict, bus_total_dict, scale=True):
        """
        assemble a list of percentage at a node, e.g. dispatch/Pmax.
        Arguments:
            G: the networkx object that has the rts-gmlc network structure
            bus_time_dict: a dictionary whose keys are the bus id and the values
            are the quantity of interest realized at a certain time, e.g. dispatch
            bus_total_dict: a dictionary whose keys are the bus id and the values
            are the total value of the quantity of interest e.g. Pmax.
            scale: whether we want to normalize or not. [bool]
        Returns:
            percent_list: a list of percentage in the order of the nodes' order.
        """

        percent_list = []
        for g in G.nodes():

            if bus_total_dict[g] == 0:
                # use NaN to represent nodes who dont have generators
                percent_list.append(np.nan)
            else:
                if scale:
                    percent_list.append(bus_time_dict[g] / bus_total_dict[g])
                else:
                    percent_list.append(bus_time_dict[g])

        return percent_list

    ################################################################################
    ############################ Other Plotting functions ##########################
    ################################################################################

    def line_congestion_histogram(self):

        """
        plot the histogram of line congestion (flow/flow limit) across all the
        edges.
        Arguments:
            self
        Returns:
            ax: the axes of the plot
        """

        date = self.date1
        hour = self.hour1

        df = pd.DataFrame.from_dict(
            {
                k: abs(self.line_flow_dict[k]) / self.line_flow_limit_dict[k]
                for k in self.line_flow_limit_dict.keys()
            },
            orient="index",
        )

        ax = df[0].hist(bins=10)

        ax.set_xlabel("Congestion", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_title(
            "{} Hour = {} Congestion Histogram".format(date, hour), fontsize=15
        )

        return ax

    def full_capacity_percent_histogram(self):

        """
        plot the histogram of dispatch/capacity across all the nodes.
        Arguments:
            self
        Returns:
            ax: the axes of the plot
        """

        date = self.date1
        hour = self.hour1

        full_capacity_percent_dict = {}
        for b_id in self.bus_power_delivered_dict:

            if self.bus_pmax_dict[b_id] == 0:
                continue
            else:
                full_capacity_percent_dict[b_id] = (
                    self.bus_power_delivered_dict[b_id] / self.bus_pmax_dict[b_id]
                )

        df = pd.DataFrame.from_dict(full_capacity_percent_dict, orient="index")

        ax = df[0].hist(bins=10)

        ax.set_xlabel("Congestion", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_title(
            "{} Hour = {} Full Capacity Percent Histogram".format(date, hour),
            fontsize=15,
        )

        return ax

    def plot_dispatch(
        self, ax, generator, date_list, scaled=True, label=None, linewidth=3
    ):

        """
        plot normalized dispatch as a timeseries given a generator id and a list
        of dates.
        Arguments:
            ax: an axes we want to plot the plot. [matplotlib ax]
            generator: specify a generator [str]
            date_list: a list of sequential dates we are interested in. [list]
            scaled: whether we want to scale the dispatch or not. [bool]
            label: label of the line. [str]
            linewidth: the linewidth on the plot. [float]
        Returns:
            ax: the axes of the plot
        """

        if label is None:
            label = generator

        horizon = len(date_list) * 24

        df_list = []
        for date in date_list:
            # get dispatch data
            df = get_data_given(
                self.sim1.thermal_detail_df, date=date, generator=generator
            )
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)

        # get pmax
        pmax = get_data_given(self.network.gen_param_df, generator=generator)[
            "PMax MW"
        ].values[0]

        if scaled:
            line = ax.step(
                range(1, horizon + 1),
                df["Dispatch"] / pmax,
                where="post",
                label=label,
                linewidth=linewidth,
            )
            # ax.plot(range(1,horizon + 1),df['Dispatch']/pmax,'o',color = color,alpha = 0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Scaled Dispatch", fontsize=20)
        else:
            line = ax.step(
                range(1, horizon + 1),
                df["Dispatch"],
                where="post",
                label=label,
                linewidth=linewidth,
            )
            # ax.plot(range(1,horizon + 1),df['Dispatch'],'o',color = color,alpha = 0.5)
            ax.set_ylabel("Dispatch [MW]", fontsize=20)

        color = line[0].get_color()

        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))

        ax.set_xlabel("Time [Hr]", fontsize=20)
        ax.tick_params(labelsize=20)
        # ax.set_title(''.format(bus_id,date),fontsize = 15)
        ax.legend(fontsize=20)
        ax.grid(True)

        return ax

    def plot_power_output(self, ax, generator, dates_list, title="Power Output"):

        """
        plot power output bars as a timeseries and the corresponding lmp on the
        second axis given a generator name and a list of dates.
        Arguments:
            ax: an axes we want to plot the plot. [matplotlib ax]
            generator: specify a generator [str]
            date_list: a list of sequential dates we are interested in. [list]
            title: title of the plot. [str]
        Returns:
            ax: the axes of the plot
        """

        # get the data from the dates we want
        df = self.sim1.thermal_detail_df.loc[
            self.sim1.thermal_detail_df["Date"].isin(dates_list)
        ]
        # get data given generator
        df = get_data_given(df, generator=generator)

        # get the corresponding lmp
        bus_id = int(generator[:3])
        df2 = self.sim1.bus_detail_df.loc[
            self.sim1.bus_detail_df["Date"].isin(dates_list)
        ]
        df2 = df2.loc[df2["Bus"] == self.bus_id_to_bus_name[bus_id]]

        horizon = len(df)

        y_offset = np.zeros((horizon))
        bar_width = 1

        # a list to store the lines
        ls = []

        # plot the bar
        l1 = ax.bar(
            range(1, horizon + 1),
            df["Dispatch"],
            bar_width,
            bottom=y_offset,
            alpha=0.65,
        )[0]
        y_offset += df["Dispatch"].values

        ls.append(l1)

        # plot the lmp
        ax2 = ax.twinx()
        l2 = ax2.plot(range(1, horizon + 1), df2["LMP"], "k--", linewidth=2.0)[0]
        ls.append(l2)

        ax.set_xticks(range(1, horizon + 1, 5))
        ax.set_xlabel("Time [hr]", fontsize=20)
        ax.set_ylabel("Power Dispatched [MW]", fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.tick_params(labelsize=20)
        ax.grid(True, color="k", linestyle="--", linewidth=1, alpha=0.3)

        ax2.tick_params(labelsize=20)
        ax2.set_ylim(-1, 35)
        ax2.set_ylabel("Locational Marginal\nPrice [$/MWh]", fontsize=20)

        return ls

    def plot_lmp(
        self,
        ax,
        bus_id,
        dates_list,
        title="Locational Marginal Prices",
        label=None,
        linewidth=3,
    ):

        """
        plot power output bars as a timeseries give a generator name and a list
        of dates.
        Arguments:
            ax: an axes we want to plot the plot. [matplotlib ax]
            bus_id: the id of the bus we want [int]
            date_list: a list of sequential dates we are interested in. [list]
            title: title of the plot. [str]
            label: label of the line [str]
            linewidth: the line width [float]
        Returns:
            ax: the axes of the plot
        """

        # get the corresponding lmp
        df2 = self.sim1.bus_detail_df.loc[
            self.sim1.bus_detail_df["Date"].isin(dates_list)
        ]
        df2 = df2.loc[df2["Bus"] == self.bus_id_to_bus_name[bus_id]]

        horizon = len(df2)

        # plot the lmp
        ax.plot(
            range(1, horizon + 1),
            df2["LMP"],
            linestyle="-",
            label=label,
            linewidth=linewidth,
        )

        ax.set_xticks(range(1, horizon + 1, 5))
        ax.set_xlabel("Time [hr]", fontsize=20)
        ax.set_ylabel("Locational Marginal\nPrice [$/MWh]", fontsize=20)
        ax.set_ylim(-1, 41)
        ax.set_title(title, fontsize=20)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, color="k", linestyle="--", linewidth=1, alpha=0.3)

        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))

        return ax


class ExtractedPrescientSimulationData:

    """Add methods to this class to visualize Prescient results for a single
    generator. These visualizations will NOT depend on the network structure,
    but generator parameters, such as Pmax are pulled from RTS-GMLC dataset.
    """

    summary_col = [
        "Total Dispatch [MW]",
        "Capacity Factor [%]",
        "Scaled Total Mileage [%]",
        "Total Online Time [hr]",
        "Online Fraction [%]",
        "Total Offline Time [hr]",
        "Offline Fraction [%]",
        "Average Online Time [hr]",
        "Average Offline Time [hr]",
        "Total Start-up",
        "Start-up Fraction [%]",
        "Total Shut-down",
        "Shut-down Fraction [%]",
        "Average LMP [$/MWh]",
        "Total Cost [$]",
        "Total Revenue [$]",
        "Total Profit [$]",
    ]

    gen_param_col = [
        "PMax [MW]",
        "PMin [MW]",
        "Ramp Rate [MW/hr]",
        "Min Up Time [Hr]",
        "Min Down Time [Hr]",
        "Marginal Cost [$/MWh]",
        "No Load Cost [$/hr]",
        "Start Time Hot [Hr]",
        "Start Time Warm [Hr]",
        "Start Time Cold [Hr]",
        "Start Cost Hot [$]",
        "Start Cost Warm [$]",
        "Start Cost Cold [$]",
    ]

    def __init__(self, result_data_dir, param_data_dir):
        """
        This class reads in extracted Prescient simulation results from SNL, and
        summarize the results.
        Arguments:
            result_data_dir: directory containing results from Prescient
            simulation [string]
        """

        assert isinstance(
            result_data_dir, str
        ), "Arugment result_data_dir is the wrong type! It should be a string."

        self.result_data_dir = result_data_dir
        self.param_data_dir = param_data_dir
        self.read_result_files()

    def read_result_files(self):
        """
        This function reads full Prescient result files and set them as class
        attributes.
        """

        # thermal detail (this has the power delivered from each generator)
        self.thermal_detail_df = pd.read_csv(self.result_data_dir)

        # read the params perturbed
        with open(self.param_data_dir, "r") as f:
            self.param_data = json.load(f)

        return

    def summarize_results(
        self,
        result_num_decimals=4,
        return_numpy_arr=False,
        include_generator_param=False,
    ):
        """
        This function summarizes the results for each generator into a dataframe.
        The summary information includes:
            Energy Delivered: MWh
            Energy Averaged Price: $/MWh
            Time On: Hours
            Time On: Fraction
            Capacity Factor: MWh (delivered) / MWh (if at 100% all time)
            Startup: Number of start-up events
            Shutdown: Number of shhut-down events
            Average Time On: hours
            Average Time Off: hours
            Total Uplift Payments: $
            Number of Uplift Days
            Scaled Mileage: sum |Power(t=i+1) - Power(t=i) | / Name Plate
            Generator Characteristics
        Arguments:
            result_num_decimals: number of decimal in the result
            return_numpy_arr: if True, return the summary info as a numpy 1D
            array; Otherwise, return it as a pandas dataframe.
            include_generator_param:
            include_generator_param: if True, add generator characteristics from
            the json file [bool]
        Returns:
            if return_numpy_arr is true:
                the summay info as a numpy array
            else:
                df: the summary dataframe
        """

        # how many hours of results we have?
        total_result_horizon = len(self.thermal_detail_df)

        # get pmax
        pmax = self.param_data["p_max"]

        # total dispatch
        dispatch = self.thermal_detail_df["Dispatch"]
        total_dispatch = dispatch.sum()
        capacity_factor = total_dispatch / (pmax * total_result_horizon) * 100
        dispatch_arr = np.insert(dispatch.values, 0, 0)
        dispatch_diff = np.diff(dispatch_arr)
        total_mileage = (
            np.absolute(dispatch_diff).sum() / (pmax * total_result_horizon) * 100
        )

        # total on hours
        unit_state = self.thermal_detail_df["Unit State"]
        total_online_hour = unit_state.sum()

        online_fraction = total_online_hour / total_result_horizon * 100

        total_offline_hour = total_result_horizon - total_online_hour
        offline_fraction = 100 - online_fraction

        # find start up and shut down
        unit_state = np.insert(
            unit_state.values, 0, 0
        )  # assume generator is off before the horizon
        unit_state_diff = np.diff(unit_state)
        total_start_up = len(np.where(unit_state_diff == 1)[0])
        total_shut_down = len(np.where(unit_state_diff == -1)[0])
        start_up_fraction = total_start_up / total_result_horizon * 100
        shut_down_fraction = total_shut_down / total_result_horizon * 100

        # find on time and off time
        unit_state = self.thermal_detail_df["Unit State"]
        online_time, offline_time = PrescientSimulationData.calc_on_off_hours(
            unit_state.values
        )
        average_time_on = np.array(online_time).mean()
        average_time_off = np.array(offline_time).mean()

        # calculate the total cost using the params and result data
        production_cost = (
            dispatch.values * self.param_data["marginal_cost"]
            + unit_state.values * self.param_data["no_load_cost"]
        )
        startup_cost = self.calc_startup_cost()

        total_cost = production_cost.sum() + startup_cost.sum()

        lmp = self.thermal_detail_df["LMP"]

        revenue = lmp.values * dispatch.values
        total_revenue = revenue.sum()
        total_profit = total_revenue - total_cost
        average_price = total_revenue / total_dispatch

        # assemble a dataframe
        df = pd.DataFrame(
            [
                total_dispatch,
                capacity_factor,
                total_mileage,
                total_online_hour,
                online_fraction,
                total_offline_hour,
                offline_fraction,
                average_time_on,
                average_time_off,
                total_start_up,
                start_up_fraction,
                total_shut_down,
                shut_down_fraction,
                average_price,
                total_cost,
                total_revenue,
                total_profit,
            ],
            index=self.summary_col,
        ).T

        if include_generator_param:
            gen_param_df = self.gen_param_to_df()
            df = pd.concat([gen_param_df, df], sort=False, axis=1)

        df = df.round(
            {
                "Total Dispatch [MW]": result_num_decimals,
                "Capacity Factor [%]": result_num_decimals,
                "Scaled Total Mileage [%]": result_num_decimals,
                "Online Fraction [%]": result_num_decimals,
                "Offline Fraction [%]": result_num_decimals,
                "Start-up Fraction [%]": result_num_decimals,
                "Shut-down Fraction [%]": result_num_decimals,
                "Average LMP [$/MWh]": result_num_decimals,
                "Total Cost [$]": result_num_decimals,
                "Total Revenue [$]": result_num_decimals,
                "Total Profit [$]": result_num_decimals,
                "Average Online Time [hr]": result_num_decimals,
                "Average Offline Time [hr]": result_num_decimals,
                "PMax [MW]": result_num_decimals,
                "PMin [MW]": result_num_decimals,
                "Ramp Rate [MW/hr]": result_num_decimals,
                "Min Up Time [Hr]": result_num_decimals,
                "Min Down Time [Hr]": result_num_decimals,
                "Marginal Cost [$/MWh]": result_num_decimals,
                "No Load Cost [$/hr]": result_num_decimals,
                "Start Time Hot [Hr]": result_num_decimals,
                "Start Time Warm [Hr]": result_num_decimals,
                "Start Time Cold [Hr]": result_num_decimals,
                "Start Cost Hot [$]": result_num_decimals,
                "Start Cost Warm [$]": result_num_decimals,
                "Start Cost Cold [$]": result_num_decimals,
            }
        )

        if return_numpy_arr:
            return df.values.flatten()
        else:
            return df

    # a method to calculate start up cost
    def calc_startup_cost(self):

        """
        This function calculate the start up cost using the param_data json file
        and the unit state.
        Arguments:
            self
        Returns:
            startup_cost_arr: a 1D numpy array of start up costs. Each element is
            corresponding to each cost of each start up. It does not have the
            length of simulation horizon.
        """

        # get the unit state
        unit_state = self.thermal_detail_df["Unit State"]

        # assume all the generators have been off for 1 hour before start of horizon
        # so I insert one 0 at the beginning of the unit state
        unit_state = np.insert(unit_state.values, 0, 0)

        # calc down time
        online_time_list, offline_time_list = PrescientSimulationData.calc_on_off_hours(
            unit_state
        )

        # get startup cost profile
        # the first col is [hot start time, warm start time, cold start time]
        # the second col is the corresponding costs
        startup_cost_profile = np.array(self.param_data["startup_cost_profile"])

        startup_cost_list = []
        for offline_time in offline_time_list:

            # find out whether it is a hot, warm, or cold start
            cost_category_idx = np.searchsorted(
                startup_cost_profile[:, 0], offline_time
            )

            # cold start: down time could be arbitary long, so it can be larger
            # than the listed cold start time and the idx is == to the length
            if cost_category_idx >= len(startup_cost_profile):
                cost_category_idx = len(startup_cost_profile) - 1

            startup_cost_list.append(startup_cost_profile[cost_category_idx, 1])

        startup_cost_arr = np.array(startup_cost_list)

        return startup_cost_arr

    def gen_param_to_df(self):

        pmax = self.param_data["p_max"]
        pmin = pmax * self.param_data["p_min_multi"]
        ramp_rate = (pmax - pmin) * self.param_data["ramp_multi"]

        min_up_time = self.param_data["min_up"]
        min_dn_time = min_up_time * self.param_data["min_dn_multi"]

        marginal_cost = self.param_data["marginal_cost"]
        no_load_cost = self.param_data["no_load_cost"]

        # make sure there are 3 pairs of start up costs corresponding to hot,
        # warm and cold starts
        while len(self.param_data["startup_cost_profile"]) < 3:
            self.param_data["startup_cost_profile"].append(
                self.param_data["startup_cost_profile"][-1]
            )

        hot_start_time = self.param_data["startup_cost_profile"][0][0]
        warm_start_time = self.param_data["startup_cost_profile"][1][0]
        cold_start_time = self.param_data["startup_cost_profile"][2][0]

        hot_start_cost = self.param_data["startup_cost_profile"][0][1]
        warm_start_cost = self.param_data["startup_cost_profile"][1][1]
        cold_start_cost = self.param_data["startup_cost_profile"][2][1]

        # assemble a dataframe
        df = pd.DataFrame(
            [
                pmax,
                pmin,
                ramp_rate,
                min_up_time,
                min_dn_time,
                marginal_cost,
                no_load_cost,
                hot_start_time,
                warm_start_time,
                cold_start_time,
                hot_start_cost,
                warm_start_cost,
                cold_start_cost,
            ],
            index=self.gen_param_col,
        ).T

        return df
