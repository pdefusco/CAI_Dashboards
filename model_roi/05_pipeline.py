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

import string
import random
from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from datetime import datetime
from pprint import pprint
import json, secrets, os, time
import mlflow
from mlops import ModelDeployment
from cmlapi.utils import Cursor
from mlops import ModelDeployment

# current date and time
now = datetime.now()

timestamp = datetime.timestamp(now)

try:
    client = cmlapi.default_client()
except ValueError:
    print("Could not create a client. If this code is not being run in a CAI session, please include the keyword arguments \"url\" and \"cml_api_key\".")

session_id = "".join([random.choice(string.ascii_lowercase) for _ in range(6)])
session_id

cluster = os.getenv("CDSW_DOMAIN")

# Set correlation factor
x = random.randint(1, 5)

# Set project ID
project_id = os.environ["CDSW_PROJECT_ID"]

# Create the DATAGEN JOB
# Create a job. We will create dependent/children jobs of this job, so we call this one a "grandparent job". The parameter "runtime_identifier" is needed if this is running in a runtimes project.
datagen_job_body = cmlapi.CreateJobRequest(
    project_id = project_id,
    name = "datagen_"+session_id,
    script = "model_roi/00_datagen.py",
    cpu = 4.0,
    memory = 8.0,
    runtime_identifier = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers = ["spark351-24.1-h1"],
    environment = {
                    "x": str(x)
                    }
)
datagen_job = client.create_job(datagen_job_body, project_id)

# Run the DATAGEN Job
datagen_jobrun_body = cmlapi.CreateJobRunRequest(project_id, datagen_job.id)
datagen_job_run = client.create_job_run(datagen_jobrun_body, project_id, datagen_job.id)

### Poll for DATAGEN Job Status

try:
    # Gets a job run.
    api_response = api_instance.get_job_run(project_id, datagen_job.id, datagen_job_run.id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->get_job_run: %s\n" % e)

if api_response.status == "complete":
    continue
else:
    pass #just wait here

# Create the TRAIN MODEL JOB
# Create a job. We will create dependent/children jobs of this job, so we call this one a "grandparent job". The parameter "runtime_identifier" is needed if this is running in a runtimes project.
train_model_job_body = cmlapi.CreateJobRequest(
    project_id = project_id,
    name = "train_model_"+session_id,
    script = "model_roi/01_train_model.py",
    cpu = 4.0,
    memory = 8.0,
    runtime_identifier = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers = ["spark351-24.1-h1"]
)
train_model_job = client.create_job(train_model_job_body, project_id)

# Run the TRAIN MODEL Job
train_model_jobrun_body = cmlapi.CreateJobRunRequest(project_id, train_model_job.id)
train_model_job_run = client.create_job_run(train_model_jobrun_body, project_id, datagen_job.id)

### Poll for DATAGEN Job Status
try:
    # Gets a job run.
    api_response = api_instance.get_job_run(project_id, train_model_job.id, train_model_job_run.id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->get_job_run: %s\n" % e)

if api_response.status == "complete":
    continue
else:
    pass #just wait here

# Create the MODEL DEPLOYMENT
deployment = ModelDeployment(client, projectId, username, experimentName, experimentId)

modelName = "Model-CLF"

createModelResponse = deployment.createModel(projectId, modelName, modelId)
modelCreationId = createModelResponse.id

runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5" #Modify as needed
createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId)
modelBuildId = createModelBuildResponse.id

deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)

try:
    # List model deployments, optionally filtered, sorted, and paginated.
    api_response = api_instance.list_model_deployments(project_id, modelCreationId, modelBuildId)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_model_deployments: %s\n" % e)

if api_response.status == "complete":
    continue
else:
    pass #just wait here

# Create the SIMULATION JOB
simulation_job_body = cmlapi.CreateJobRequest(
    project_id = project_id,
    name = "simulation_"+session_id,
    script = "model_roi/03_simulation.py",
    cpu = 4.0,
    memory = 8.0,
    runtime_identifier = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers = ["spark351-24.1-h1"],
    parent_job_id = sparkgen_1_job.id
)
simulation_job = client.create_job(simulation_job_body, project_id)

# Run the Simulation Jobs
simulation_jobrun_body = cmlapi.CreateJobRunRequest(project_id, simulation_job.id)
simulation_job_run = client.create_job_run(simulation_jobrun_body, project_id, simulation_job.id)

### Poll for Simulation Job Status
try:
    # Gets a job run.
    api_response = api_instance.get_job_run(project_id, simulation_job.id, simulation_job_run.id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->get_job_run: %s\n" % e)

if api_response.status == "complete":
    continue
else:
    pass #just wait here

### Create Application
application_body = cmlapi.CreateApplicationRequest(
    project_id = project_id,
    name = "Live Model ROI Dashboard",
    subdomain = "Org",
    description = "Continuously Updating Model ROI Dashboard",
    script = "model_roi/03_simulation.py",
    cpu = 4.0,
    memory = 8.0,
    runtime_identifier = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    bypass_authentication = True
    )

try:
    # Create an application and implicitly start it immediately.
    api_response = api_instance.create_application(application_body, project_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_application: %s\n" % e)
