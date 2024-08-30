import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LostImagesComponent } from './lost-images.component';

describe('LostImagesComponent', () => {
  let component: LostImagesComponent;
  let fixture: ComponentFixture<LostImagesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [LostImagesComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(LostImagesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
