import glob
import os
from prefect.testing.utilities import prefect_test_harness
from prefect_flows.flows.manual_meeting_summary_flow import manual_meeting_summary_flow


def test_exec():
    with prefect_test_harness():

        manual_meeting_summary_flow()

if __name__ == "__main__":
    test_exec()

"""
"""
