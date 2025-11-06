#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import time
from pprint import pprint
import cmlapi
from cmlapi.rest import ApiException

class PipelineUtil:

    '''Class to Manage Pipeline'''

    def __init__(self, project_id, username):
        self.project_id = project_id
        self.username = username
        self.client = cmlapi.default_client()

    def poll_job_status(job_id, job_run_id, poll_interval=10, timeout=600):
        """
        Polls a CML job run until it completes, fails, or times out.

        Args:
            client: An instance of the CMLServiceApi.
            project_id (str): The CML project ID.
            job_id (str): The job ID.
            job_run_id (str): The job run ID.
            poll_interval (int): Time (in seconds) between status checks.
            timeout (int): Maximum time (in seconds) to wait before timing out.

        Returns:
            The final job run object (api_response) if successful.

        Raises:
            TimeoutError: If the job doesn't complete within the timeout.
            ApiException: If the API call fails.
        """
        start_time = time.time()

        while True:
            try:
                api_response = self.client.get_job_run(self.project_id, job_id, job_run_id)
                status = api_response.status.lower()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Job run {job_run_id} status: {status}")

                if status in ["complete", "failed", "cancelled"]:
                    print("Job finished with status:", status.upper())
                    pprint(api_response)
                    return api_response

            except ApiException as e:
                print(f"Exception when calling CMLServiceApi->get_job_run: {e}")
                print("Retrying in 10 seconds...")
                time.sleep(10)
                continue

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Polling timed out after {timeout} seconds for job run {job_run_id}.")

            # Wait before the next check
            time.sleep(poll_interval)


    def create_and_run_job(
        x=None,
        script_path: str,
        job_name_prefix: str = "datagen_",
        cpu: float,
        memory: float,
        runtime_identifier: str = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
        runtime_addon_identifiers: list = None
    ):
        """
        Creates and runs a data generation job in Cloudera Machine Learning (CML).

        Parameters
        ----------
        client : object
            The initialized CML API client.
        cmlapi : module
            The CML API Python module (typically imported as `import cmlapi`).
        project_id : str
            The CML project ID where the job will be created.
        x : any, optional
            Optional environment variable 'x'. If not provided, it's omitted.
        script_path : str, optional
            Path to the Python script to execute in the job.
        job_name_prefix : str, optional
            Prefix for the generated job name.
        cpu : float, optional
            Number of CPUs to allocate for the job.
        memory : float, optional
            Amount of memory (in GB) to allocate for the job.
        runtime_identifier : str, optional
            The CML runtime identifier to use.
        runtime_addon_identifiers : list, optional
            List of runtime addon identifiers. Defaults to ["spark351-24.1-h1"].

        Returns
        -------
        dict
            A dictionary containing the created job and job run objects.
        """
        if runtime_addon_identifiers is None:
            runtime_addon_identifiers = ["spark351-24.1-h1"]

        environment = {}
        if x is not None:
            environment["x"] = str(x)

        # Create the DATAGEN Job
        datagen_job_body = cmlapi.CreateJobRequest(
            project_id=self.project_id,
            name=f"{job_name_prefix}",
            script=script_path,
            cpu=cpu,
            memory=memory,
            runtime_identifier=runtime_identifier,
            runtime_addon_identifiers=runtime_addon_identifiers,
            environment=environment if environment else None
        )

        datagen_job = self.client.create_job(datagen_job_body, self.project_id)

        # Run the DATAGEN Job
        datagen_jobrun_body = cmlapi.CreateJobRunRequest(self.project_id, datagen_job.id)
        datagen_job_run = client.create_job_run(datagen_jobrun_body, self.project_id, datagen_job.id)

        return {
            "job": datagen_job,
            "job_run": datagen_job_run
        }

    def wait_for_model_build_complete(
        model_creation_id: str,
        model_build_id: str,
        poll_interval: int = 10,
        timeout: int = 1800
    ):
        """
        Polls for the status of a CML model build until it reaches 'complete' or times out.

        Parameters
        ----------
        self.client : object
            An instance of the initialized CML API service client (e.g., cmlapi.CMLServiceApi(client)).
        project_id : str
            The ID of the CML project containing the model.
        model_creation_id : str
            The unique ID of the model creation.
        model_build_id : str
            The unique ID of the model build.
        poll_interval : int, optional
            How many seconds to wait between polling attempts. Default is 10 seconds.
        timeout : int, optional
            Maximum number of seconds to wait before timing out. Default is 1800 seconds (30 minutes).

        Returns
        -------
        dict
            The final API response once the build reaches 'complete', or None if it times out or fails.
        """
        start_time = time.time()

        print(f"Waiting for model build {model_build_id} to reach 'complete' state...")

        while True:
            try:
                # Query model deployment status
                api_response = self.client.list_model_deployments(
                    self.project_id, model_creation_id, model_build_id
                )
                pprint(api_response)

                # If build is complete, break out
                if getattr(api_response, "status", "").lower() == "complete":
                    print("Model build has reached 'complete' state.")
                    return api_response

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print("Timed out waiting for model build to complete.")
                    return None

                # Wait before polling again
                time.sleep(poll_interval)

            except Exception as e:
                print(f"Exception when calling CMLServiceApi->list_model_deployments: {e}")
                time.sleep(poll_interval)


        def wait_for_model_deployment_status(
            model_id: str,
            build_id: str,
            desired_status: str = "deployed",
            poll_interval: int = 10,
            timeout: int = 1800,
            search_filter: str = None,
            page_size: int = None,
            page_token: str = None,
            sort: str = None
        ):
            """
            Polls for a model deployment to reach a desired status in Cloudera Machine Learning (CML).

            Parameters
            ----------
            self.client : object
                An instance of the initialized CML API service client (e.g., cmlapi.CMLServiceApi(client)).
            project_id : str
                The ID of the CML project containing the model.
            model_id : str
                The ID of the model to check deployment status for.
            build_id : str
                The ID of the model build associated with the deployment.
            desired_status : str, optional
                The deployment status to wait for (default is "deployed").
            poll_interval : int, optional
                Number of seconds to wait between polling attempts (default is 10 seconds).
            timeout : int, optional
                Maximum number of seconds to wait before giving up (default is 1800 seconds).
            search_filter : str, optional
                Optional filter string to narrow down deployments.
            page_size : int, optional
                Page size for the deployment list response.
            page_token : str, optional
                Page token for paginated responses.
            sort : str, optional
                Sort key for ordering results.

            Returns
            -------
            dict or None
                The final API response when the deployment reaches the desired status,
                or None if the operation times out.
            """
            start_time = time.time()

            print(f"Waiting for model deployment (build ID: {build_id}) to reach '{desired_status}' state...")

            while True:
                try:
                    # List model deployments (optionally with filters and pagination)
                    api_response = self.client.list_model_deployments(
                        self.project_id,
                        model_id,
                        build_id,
                        search_filter=search_filter,
                        page_size=page_size,
                        page_token=page_token,
                        sort=sort
                    )

                    pprint(api_response)

                    # Extract the deployment status
                    # (Handle both attribute-style and dict-style responses)
                    current_status = getattr(api_response, "status", None)
                    if current_status is None and hasattr(api_response, "deployments"):
                        # Try to read from deployments list if present
                        deployments = getattr(api_response, "deployments", [])
                        if deployments and hasattr(deployments[0], "status"):
                            current_status = deployments[0].status

                    # Check if status matches desired state
                    if current_status and current_status.lower() == desired_status.lower():
                        print(f"Model deployment has reached '{desired_status}' state.")
                        return api_response

                    # Timeout check
                    if time.time() - start_time > timeout:
                        print("Timed out waiting for model deployment to reach desired status.")
                        return None

                    # Wait before polling again
                    time.sleep(poll_interval)

                except ApiException as e:
                    print(f"API Exception when calling CMLServiceApi->list_model_deployments: {e}")
                    time.sleep(poll_interval)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    time.sleep(poll_interval)


    def create_cml_application(
        name: str,
        subdomain: str,
        script: str,
        description: str = "CML Application",
        cpu: float = 4.0,
        memory: float = 8.0,
        runtime_identifier: str = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
        bypass_authentication: bool = True
    ):
        """
        Creates a new Cloudera Machine Learning (CML) Application.

        Parameters
        ----------
        api_instance : object
            Instance of the initialized CML API service client (e.g., cmlapi.CMLServiceApi(client)).
        cmlapi : module
            The CML API Python module (typically imported as `import cmlapi`).
        project_id : str
            The ID of the project in which to create the application.
        name : str
            The display name of the application.
        subdomain : str
            The subdomain for the application's public endpoint.
        script : str
            Path to the script to execute (e.g., "model_roi/03_simulation.py").
        description : str, optional
            Description of the application (default: "CML Application").
        cpu : float, optional
            Number of CPUs to allocate (default: 4.0).
        memory : float, optional
            Memory to allocate in GB (default: 8.0).
        runtime_identifier : str, optional
            Runtime image to use for the application.
        bypass_authentication : bool, optional
            Whether to bypass authentication for the application (default: True).

        Returns
        -------
        dict or None
            The created application’s API response, or None if creation fails.
        """
        application_body = cmlapi.CreateApplicationRequest(
            project_id=self.project_id,
            name=name,
            subdomain=subdomain,
            description=description,
            script=script,
            cpu=cpu,
            memory=memory,
            runtime_identifier=runtime_identifier,
            bypass_authentication=bypass_authentication
        )

        try:
            print(f"Creating application '{name}' in project {self.project_id}...")
            api_response = self.client.create_application(application_body, project_id)
            pprint(api_response)
            print("Application created successfully.")
            return api_response

        except ApiException as e:
            # Handle "already exists" or conflict errors gracefully
            error_message = str(e).lower()
            if "already exists" in error_message or "409" in error_message:
                print(f"Application '{name}' already exists — skipping creation.")
                return None
            else:
                print(f"Exception when calling CMLServiceApi->create_application: {e}\n")
                return None
