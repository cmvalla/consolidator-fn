# Consolidator Function

This directory contains the source code for the consolidator function.

## Architectural Choice: Cloud Build Substitutions

The `cloudbuild.yaml` file in this directory uses default substitutions that are overridden by the main Terraform configuration. This allows for a clean separation of concerns, where the `cloudbuild.yaml` defines the build steps and the main Terraform configuration provides the environment-specific values.

The default substitutions are defined in the `substitutions` block in the `cloudbuild.yaml` file. These values are placeholders that are replaced by the actual values from the main Terraform configuration during the build.

The main Terraform configuration passes the actual values for the substitutions in the `google_cloudbuild_trigger` resource for the consolidator function. This is done in the `terraform/cloud_build_triggers.tf` file.

This approach allows for the following benefits:

*   **Separation of concerns:** The `cloudbuild.yaml` file is only responsible for defining the build steps, and the main Terraform configuration is only responsible for providing the environment-specific values.
*   **Reusability:** The `cloudbuild.yaml` file can be reused for different environments without any modifications.
*   **Testability:** The `cloudbuild.yaml` file can be tested locally without having to deploy the entire infrastructure.

## Cloud Build Triggers

This function has the following Cloud Build triggers defined in the main Terraform configuration:

*   `consolidator-fn-trigger`: Triggers the deployment of the consolidator function.
*   `consolidator-fn-destroy-trigger`: Triggers the destruction of the consolidator function's resources.
