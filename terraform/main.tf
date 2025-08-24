resource "google_cloud_run_v2_service" "consolidator" {
  project  = var.project_id
  name     = "consolidator-fn"
  location = var.location
  deletion_protection = false

  template {
    service_account = var.consolidator_sa_email
    timeout         = "1800s" # 30 minutes for potentially long-running consolidations
    scaling {
      min_instance_count = 0 # Can scale to zero
      max_instance_count = 5
    }
    max_instance_request_concurrency = 50
    containers {
        image = "${var.location}-docker.pkg.dev/${var.project_id}/${var.repository_id}/${var.image_name}:${var.image_tag}"
       
        ports {
          container_port = 8080
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "REDIS_HOST"
          value = var.redis_host
        }
        env {
          name  = "REDIS_PORT"
          value = var.redis_port
        }
        
        env {
          name  = "MEMGRAPH_HOST"
          value = var.memgraph_host
        }
        env {
          name  = "MEMGRAPH_PORT"
          value = var.memgraph_port
        }
      }
  }
}

resource "google_eventarc_trigger" "consolidator_trigger" {
  project  = var.project_id
  name     = "consolidator-trigger-pubsub"
  location = var.location

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.pubsub.topic.v1.messagePublished"
  }

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.consolidator.name
      region  = var.location
    }
  }

  transport {
    pubsub {
      topic = data.google_pubsub_topic.consolidator.id
    }
  }

  service_account = var.consolidator_sa_email
}


data "google_pubsub_topic" "consolidator" {
  name = var.topic_name
  project  = var.project_id
}