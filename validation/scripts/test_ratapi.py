import os
import pathlib
import tempfile
import pytest
import sys
import numpy as np
from test_discovery import get_test_data

import ratapi

tests = list(get_test_data())
ids = [f"{t[0]}" for t in tests]


TEST_FUNC = """import numpy as np


def run_func(params, bulk_in, bulk_out, contrast):
    layers = np.array({slabs})

    # Change the units to Å
    layers[:, 1:3] = layers[:, 1:3] * 1e-6

    # Returns layers only, bulk in and bulk out added to project
    return layers[1:-1, :], params[0]
"""


@pytest.mark.parametrize("nsd", tests, ids=ids)
def test_ratapi(nsd):
    """
    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    """
    _, slabs, data = nsd

    kernel_test(slabs, data)


def kernel_test(slabs, data):
    """
    Test the reflectivity kernel for ratapi.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    """
    is_resolution_test = data.shape[1] == 4

    problem = ratapi.Project(
        name="test", model="custom layers", absorption=True
    )
    problem.scalefactors.set_fields(0, min=1, value=1, max=1)
    problem.background_parameters.set_fields(0, min=0, value=0, max=0)
    sigma = 0.021233045007200 if is_resolution_test else 0
    problem.resolution_parameters.set_fields(
        0, min=sigma, value=sigma, max=sigma
    )

    with np.printoptions(threshold=sys.maxsize):
        slab_string = np.array2string(slabs, separator=", ")

    # Write a custom file that conatains the slabs.
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as f:
        f.write(TEST_FUNC.format(slabs=slab_string))
        f.flush()

        sub_rough = slabs[-1, -1]

        # Change the units to Å
        bulk_in = slabs[0, 1] * 1e-6
        bulk_out = slabs[-1, 1] * 1e-6

        problem.parameters.set_fields(
            0, min=sub_rough, value=sub_rough, max=sub_rough
        )
        problem.bulk_in.set_fields(
            0,
            name="Bulk In",
            min=bulk_in,
            value=bulk_in,
            max=bulk_in,
            fit=False,
        )
        problem.bulk_out.set_fields(
            0,
            name="Bulk Out",
            min=bulk_out,
            value=bulk_out,
            max=bulk_out,
            fit=False,
        )

        if is_resolution_test:
            # Using linearly spaced points similar to the ref1d test
            n_points = 10000

            argmin = np.argmin(data[:, 0])
            argmax = np.argmax(data[:, 0])
            data_more_points = np.zeros((n_points, 3))

            data_more_points[:, 0] = np.linspace(
                data[argmin, 0] - 3.5 * data[argmin, 3],
                data[argmax, 0] + 3.5 * data[argmax, 3],
                n_points,
            )
            problem.data.append(name="Data", data=data_more_points)
        else:
            # RAT expects minimum of 3 columns in data so add zero column.
            data = np.column_stack((data, np.zeros(len(data))))
            problem.data.append(name="Data", data=data)

        problem.custom_files.append(
            name="Model",
            filename=pathlib.Path(f.name).name,
            language="python",
            path=pathlib.Path(f.name).parent,
            function_name="run_func",
        )

        problem.contrasts.append(
            name="ORSO Contrast",
            background="Background 1",
            background_action="add",
            resolution="Resolution 1",
            scalefactor="Scalefactor 1",
            bulk_out="Bulk Out",
            bulk_in="Bulk In",
            data="Data",
            resample=False,
            model=["Model"],
        )

        controls = ratapi.Controls(display="off")
        problem, results = ratapi.run(problem, controls)

        if is_resolution_test:
            ref = np.interp(
                data[:, 0],
                data_more_points[:, 0],
                results.reflectivity[0][:, 1],
            )
            np.testing.assert_allclose(ref, data[:, 1], rtol=0.033)
        else:
            np.testing.assert_allclose(
                results.reflectivity[0][:, 1], data[:, 1], rtol=8e-5
            )

        f.close()
        os.remove(f.name)


if __name__ == "__main__":
    for nsd in tests:
        test_ratapi(nsd)
