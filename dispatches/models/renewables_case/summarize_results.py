import os
import json
import pandas as pd


def _convert_sim_opt_txt_to_dict(file_path):

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")

    with open(file_path) as f:
        for l in f:
            new_line = l.replace("'", '"')
            return json.loads(new_line)


def _summarize(sim_id, result_dir, gen_detail, bus_name, gen_name):

    df = pd.read_csv(os.path.join(result_dir, gen_detail))
    df = df.loc[df["Generator"] == gen_name]
    df["Time Index"] = range(len(df))
    df.rename(columns={"Output": "Dispatch", "Output DA": "Dispatch DA"}, inplace=True)

    bus_df = pd.read_csv(os.path.join(result_dir, "bus_detail.csv"))
    bus_df = bus_df.loc[bus_df["Bus"] == bus_name]
    bus_df["Time Index"] = range(len(bus_df))

    df = df.merge(bus_df, how="left", left_on="Time Index", right_on="Time Index")

    df["Revenue DA"] = df["Dispatch DA"] * df["LMP DA"]
    df["Revenue RT"] = (df["Dispatch"] - df["Dispatch DA"]) * df["LMP"]
    df["Total Revenue"] = df["Revenue DA"] + df["Revenue RT"]

    df = df[["Dispatch", "Dispatch DA", "Revenue DA", "Revenue RT", "Total Revenue"]]

    summary = df.sum().to_dict()
    summary["sim_id"] = sim_id

    return summary


def summarize_sim_results(sim_id, result_dir, bus_name="Carter", gen_name="309_WIND_1"):

    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"{result_dir} does not exist!")

    # find the participation mode
    opt_dict = _convert_sim_opt_txt_to_dict(os.path.join(result_dir, "sim_options.txt"))

    if opt_dict["participation_mode"] == "Bid":
        gen_detail = "thermal_detail.csv"
    elif opt_dict["participation_mode"] == "SelfSchedule":
        gen_detail = "renewables_detail.csv"

    return _summarize(sim_id, result_dir, gen_detail, bus_name, gen_name)


def summarize_and_concat(
    sim_id_list, result_dir_generator, bus_name="Carter", gen_name="309_WIND_1"
):

    opt_summary_list = []
    result_summary_list = []

    for sim_id in sim_id_list:

        print(f"Summarizing simulation {sim_id}...")

        result_dir = result_dir_generator(sim_id)

        # summarize sim options
        file_path = os.path.join(result_dir, "sim_options.txt")

        try:
            d = _convert_sim_opt_txt_to_dict(file_path)
        except FileNotFoundError as ex:
            print(f"Simulation {sim_id} does not have sim_options.txt!")
        else:
            opt_summary_list.append(d)

        # summarize results
        try:
            d = summarize_sim_results(
                sim_id=sim_id,
                result_dir=result_dir,
                bus_name=bus_name,
                gen_name=gen_name,
            )
        except FileNotFoundError as ex:
            print(f"Simulation {sim_id} does not have a result dir!")
        else:
            result_summary_list.append(d)

    option_summary_df = pd.DataFrame(opt_summary_list).set_index("sim_id")
    result_summary_df = pd.DataFrame(result_summary_list).set_index("sim_id")

    return option_summary_df, result_summary_df


if __name__ == "__main__":

    result_dir_generator = lambda sim_id: os.path.join(
        ".", f"batch_3_sim_{sim_id}_results"
    )
    sim_id_list = list(range(400))

    option_summary_df, result_summary_df = summarize_and_concat(
        sim_id_list, result_dir_generator, bus_name="Carter", gen_name="309_WIND_1"
    )

    # write to csv
    option_summary_df.to_csv("batch_3_sim_option_summary.csv")
    result_summary_df.to_csv("batch_3_sim_result_summary.csv")
