# Release Notes

- 2022.11.16
  - model:
    - name: [`VMC_v1`](https://drive.google.com/file/d/1-DhSsvfVjGxdcna8rL0GdR_M_8zIrgLe/view?usp=drivesdk)
    - architecture: `shufflenet_v2_x2_0`
    - optimizer: `SGD` - 15e
    - transform: `RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`, `RandomRotation`
    - optimization technique: `weighted_loss`, `weighted_sampling`
  - data:
    - name: [`vn_veri_wild_v1`](https://drive.google.com/uc?id=1P6zYXMKeekWVj5CI0921F5x8ECANQBMO)
    - source: `VERI-Wild dataset`
    - detail: 155363 images (124290 train imgs, 31073 val imgs); 19 vehicle makes in Vietnam; using `undersampling` technique.
  - metric:
    - acc: 98.18%
    - f1: 98.17%
