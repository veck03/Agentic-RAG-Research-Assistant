from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/ocean/sample_ocean_data.csv")


class OceanDataTool:
    """
    Tool for querying structured oceanographic data.

    The tool performs numerical operations using pandas
    instead of asking the LLM to calculate values itself.
    """

    def __init__(self, data_file=DATA_FILE):

        self.data_file = data_file

        self.df = pd.read_csv(
            self.data_file,
            parse_dates=["date"]
        )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    def dataset_info(self):

        return {
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "start_date": str(
                self.df["date"].min().date()
            ),
            "end_date": str(
                self.df["date"].max().date()
            ),
            "depths": sorted(
                self.df["depth_m"].unique().tolist()
            ),
        }

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    def filter_data(
        self,
        start_date=None,
        end_date=None,
        min_lat=None,
        max_lat=None,
        min_lon=None,
        max_lon=None,
        depth_m=None
    ):

        data = self.df.copy()

        if start_date is not None:
            data = data[
                data["date"] >= pd.to_datetime(start_date)
            ]

        if end_date is not None:
            data = data[
                data["date"] <= pd.to_datetime(end_date)
            ]

        if min_lat is not None:
            data = data[
                data["latitude"] >= min_lat
            ]

        if max_lat is not None:
            data = data[
                data["latitude"] <= max_lat
            ]

        if min_lon is not None:
            data = data[
                data["longitude"] >= min_lon
            ]

        if max_lon is not None:
            data = data[
                data["longitude"] <= max_lon
            ]

        if depth_m is not None:
            data = data[
                data["depth_m"] == depth_m
            ]

        return data

    # --------------------------------------------------------
    # Average temperature
    # --------------------------------------------------------

    def average_temperature(self, **filters):

        data = self.filter_data(**filters)

        if data.empty:
            return None

        return float(
            data["temperature_c"].mean()
        )

    # --------------------------------------------------------
    # Maximum temperature
    # --------------------------------------------------------

    def maximum_temperature(self, **filters):

        data = self.filter_data(**filters)

        if data.empty:
            return None

        row = data.loc[
            data["temperature_c"].idxmax()
        ]

        return {
            "temperature_c": float(
                row["temperature_c"]
            ),
            "date": str(
                row["date"].date()
            ),
            "latitude": float(
                row["latitude"]
            ),
            "longitude": float(
                row["longitude"]
            ),
            "depth_m": float(
                row["depth_m"]
            ),
        }

    # --------------------------------------------------------
    # Minimum temperature
    # --------------------------------------------------------

    def minimum_temperature(self, **filters):

        data = self.filter_data(**filters)

        if data.empty:
            return None

        row = data.loc[
            data["temperature_c"].idxmin()
        ]

        return {
            "temperature_c": float(
                row["temperature_c"]
            ),
            "date": str(
                row["date"].date()
            ),
            "latitude": float(
                row["latitude"]
            ),
            "longitude": float(
                row["longitude"]
            ),
            "depth_m": float(
                row["depth_m"]
            ),
        }

    # --------------------------------------------------------
    # Temperature trend
    # --------------------------------------------------------

    def yearly_average_temperature(self):

        result = (
            self.df
            .groupby(
                self.df["date"].dt.year
            )["temperature_c"]
            .mean()
            .reset_index()
        )

        result.columns = [
            "year",
            "average_temperature_c"
        ]

        return result.to_dict(
            orient="records"
        )


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    tool = OceanDataTool()

    print("\n===== DATASET INFO =====")
    print(tool.dataset_info())

    print("\n===== AVERAGE TEMPERATURE =====")

    average = tool.average_temperature(
        depth_m=0
    )

    print(
        f"Average surface temperature: "
        f"{average:.2f} °C"
    )

    print("\n===== MAXIMUM TEMPERATURE =====")

    maximum = tool.maximum_temperature()

    print(maximum)

    print("\n===== MINIMUM TEMPERATURE =====")

    minimum = tool.minimum_temperature()

    print(minimum)

    print("\n===== YEARLY AVERAGES =====")

    yearly = tool.yearly_average_temperature()

    for row in yearly:
        print(row)