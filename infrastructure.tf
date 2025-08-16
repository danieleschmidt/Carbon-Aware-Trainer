
# Carbon-Aware-Trainer Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Multi-region EKS clusters
resource "aws_eks_cluster" "carbon_trainer" {
  for_each = toset(var.regions)
  
  name     = "carbon-trainer-${each.value}"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = module.vpc[each.value].private_subnets
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
}

# Auto-scaling node groups
resource "aws_eks_node_group" "carbon_trainer" {
  for_each = toset(var.regions)
  
  cluster_name    = aws_eks_cluster.carbon_trainer[each.value].name
  node_group_name = "carbon-trainer-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = module.vpc[each.value].private_subnets

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  instance_types = ["t3.medium"]
  
  remote_access {
    ec2_ssh_key = var.key_name
  }
}

# Global load balancer
resource "aws_lb" "global" {
  name               = "carbon-trainer-global-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = false
}

variable "regions" {
  description = "AWS regions for deployment"
  type        = list(string)
  default     = ["us-west-2", "us-east-1", "eu-west-1"]
}
