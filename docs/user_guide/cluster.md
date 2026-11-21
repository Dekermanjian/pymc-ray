# Running on a cluster

This page explains how to make your model functions available to a Ray cluster, and how to start and stop a Ray cluster on Microsoft Azure.

## Model functions on a cluster

Model functions are pickled to the workers. Lambdas and closures defined in your script work as they are. Functions imported from your own modules must be importable on the
workers: use `ray.init(runtime_env={"working_dir": "."})` or `py_modules=[...]`.

## Infrastructure: a Ray cluster on Azure

[infrastructure/azure-config.yaml](https://github.com/Dekermanjian/pymc-ray/blob/main/infrastructure/azure-config.yaml) describes a three-node cluster in `eastus2`: one head node and two workers, all `Standard_F16as_v6` (16 vCPUs, one physical core per vCPU). Every node gets Python 3.13, Ray 2.59.0, PyMC 6.3.2, PyTensor 3.3.4, and this repo installed with its `test` extras, which include nutpie. The head gives Ray 12 of its 16 cores and keeps 4 free for the chains, which run on the head, so Ray sees 44 CPUs in total.

The VMs are billed for as long as they exist, so bring the cluster down when you are done. Run every command below from the repo root.

### Prerequisites

- **The Azure CLI**, signed in to the subscription that will pay for the cluster. Ray uses the CLI's login.

  ```bash
  az login
  az account show --query "{name: name, id: id}" -o table # check that this is the right subscription
  ```

- **vCPU quota** for 48 vCPUs of the Fasv6 family in `eastus2` (3 nodes × 16 vCPUs), and for 48 regional vCPUs in total. New subscriptions often have less. Request an increase under *Quotas* in the Azure portal if needed.

  ```bash
  az vm list-usage --location eastus2 -o table | grep -iE "fasv6|total regional"
  ```

- **The Azure SDK packages that Ray's launcher uses**, in the `pymc-ray` environment. These are the same versions the head node installs:

  ```bash
  conda activate pymc-ray
  pip install azure-core==1.35.0 azure-identity==1.23.1 azure-mgmt-compute==35.0.0 azure-mgmt-network==29.0.0 azure-mgmt-resource==24.0.0 azure-common==1.1.28 msrest==0.7.1 msrestazure==0.6.4.post1
  ```

- **Ray 2.59.0 and Python 3.13 on your machine.** Ray Client refuses to connect when the Ray version or the Python minor version differs from the cluster's. Check with `ray --version`, and run `pip install "ray[default]==2.59.0"` if it shows another version.

- **`envsubst`**, which comes with gettext (`brew install gettext` on macOS).

### Bring the cluster up

The config holds the subscription id as `${AZURE_SUBSCRIPTION_ID}`, which Ray does not expand. Render the id into a copy of the config first. The rendered copy is git-ignored, and every later command uses it.

```bash
export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
envsubst '$AZURE_SUBSCRIPTION_ID' < infrastructure/azure-config.yaml > infrastructure/azure-config.rendered.yaml
ray up -y infrastructure/azure-config.rendered.yaml
```

`ray up` creates the resource group `ray-cluster-rg`, with a virtual network, a network security group that lets in SSH only, and a managed identity. It then starts the head node, copies this repo to it, and installs everything, which takes several minutes. Then the head launches the two workers itself, which takes a few more minutes. Wait for both workers before you create a `ShardedModel`, because actors created while workers are still starting all land on the head:

```bash
ray exec infrastructure/azure-config.rendered.yaml "ray status" # repeat until 2 ray.worker.default nodes are active
```

After you change the code locally, rerun `ray up` with the same config to copy it to the cluster again.

### Connect with Ray Client (port 10001)

The network security group lets in SSH only, so you reach the head node's Ray Client server (port 10001) through an SSH tunnel. `ray attach` opens a shell on the head node and forwards the port for as long as that shell stays open:

```bash
ray attach infrastructure/azure-config.rendered.yaml -p 10001
```

Leave that terminal open. In another terminal, point Ray at the tunnel, then start Python or Jupyter from that shell:

```bash
export RAY_ADDRESS=ray://localhost:10001
```

`ray.init()` reads `RAY_ADDRESS`, and the example notebooks do too. Code sent through Ray Client runs on the cluster's Python, so pass a `runtime_env` for anything it is missing, as the [ShardedModel tutorial](../tutorials/sharded_tutorial.ipynb) does.

### Open the Ray dashboard

```bash
ray dashboard infrastructure/azure-config.rendered.yaml
```

This forwards the head node's dashboard (port 8265) to http://localhost:8265 until you stop it with Ctrl-C. If local port 8265 is already in use, for example by a tunnel to another cluster, pick another local port with `--port 8266` and open http://localhost:8266. The same address is the one to pass to `ray job submit --address` (see [Measuring on your own hardware](performance.md#measuring-on-your-own-hardware)).

To forward both ports from one terminal, run `ray attach infrastructure/azure-config.rendered.yaml -p 10001 -p 8265` instead.

### Bring the cluster down

```bash
ray down -y infrastructure/azure-config.rendered.yaml
```

This deletes the VMs together with their disks, network interfaces and public IP addresses. With `cache_stopped_nodes: false` in the config, nodes are deleted rather than stopped. The resource group stays, with its virtual network, network security group and managed identity, so that the next `ray up` can reuse them. These do not cost anything on their own.

### Delete every Azure resource

To be sure nothing is left that could be billed, delete the resource group. This removes everything Ray created in it, including any VMs that are still running, so it also works if `ray down` failed.

```bash
az group delete --name ray-cluster-rg --yes
az group exists --name ray-cluster-rg # prints false once it is gone
```

Azure may also have created a `NetworkWatcherRG` resource group when it created the virtual network. Network Watcher is free, and other virtual networks in the subscription may use it, so leave it in place unless you know that nothing else needs it.
