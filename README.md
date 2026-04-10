Under construction!

# Benchmark Yeast Sequence-to-Expression models
> [!NOTE]  
> In case of any questions, reach out to mail@timonschneider.de - always happy to help!


This is a collection of (hopfully) easy-to-use datasets and scripts to benchmark models that predict some type of gene expression from DNA sequence. 

Our core test datasets are:
1. de Boer/Rafi et al. MPRA
2. Shalem et al. MPRA
3. ...

## Usage

```python 
# TODO
```

## Contact
Reach out to mail@timonschneider.de in case you have questions/need help/want to chat.

## Roadmap
- [ ] Add Rafi's MPRA eval
- [ ] Add eQTL Caudal eval
- [ ] Add eQTL Kita eval

## Notes
- What are the constraints: 
    - PyTorch first - we optimize our code so it's easy to benchmark pytorch models
- how to force/encourage leakage quantification
 - HashFrag?
- how to set up a good benchmark
    - needs to include all previous benchmarks that are relevant
- which benchmarks are already out there?
    - Shorkie
    - Yorzoi
    - SpeciesLM
    - Karollus benchmark paper
    - bend?
    - NTv3?
- which models to benchmark? 
    - Baseline?
        - naive U-Net trained on Yorzoi data
    - Shorkie
    - Yorzoi
    - SpeciesLM
    - ...? 
- make list of datasets from Yorzoi, SpeciesLM and Shorkie paper
- would be fun to have a live leaderboard?