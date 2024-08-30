// shared.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SharedService {
  private artistNameSource = new BehaviorSubject<string>('');
  private similaritySource = new BehaviorSubject<string>('');
  private closestImageIdSource = new BehaviorSubject<string>('');
  private paintingNameSource = new BehaviorSubject<string>('');
  private fileNameSource = new BehaviorSubject<string>('');
  private closestImageDataSource = new BehaviorSubject<string>('');
  private artistImagesSource = new BehaviorSubject<any[]>([]);
  private closeImagesSource = new BehaviorSubject<any[]>([]);
  private closenessThresoldSource = new BehaviorSubject<string>('');
  private averageDistanceSource = new BehaviorSubject<string>('');

  artistName$ = this.artistNameSource.asObservable();
  similarity$ = this.similaritySource.asObservable();
  closestImageId$ = this.closestImageIdSource.asObservable();
  paintingName$ = this.paintingNameSource.asObservable();
  fileName$ = this.fileNameSource.asObservable();
  closestImageData$ = this.closestImageDataSource.asObservable();
  artistImages$ = this.artistImagesSource.asObservable();
  closeImages$ = this.closeImagesSource.asObservable();
  closenessThreshold$ = this.closenessThresoldSource.asObservable();
  averageDistance$ = this.averageDistanceSource.asObservable();


  // Lost Paintings variables:
  private artistNameLPSource = new BehaviorSubject<string>('');
  private similarityLPSource = new BehaviorSubject<string>('');
  private closestImageIdLPSource = new BehaviorSubject<string>('');
  private paintingNameLPSource = new BehaviorSubject<string>('');
  private fileNameLPSource = new BehaviorSubject<string>('');
  private closestImageDataLPSource = new BehaviorSubject<string>('');
  private closeImagesLPSource = new BehaviorSubject<any[]>([]);
  private closenessThresholdLPSource = new BehaviorSubject<string>('');

  artistNameLP$ = this.artistNameLPSource.asObservable();
  similarityLP$ = this.similarityLPSource.asObservable();
  closestImageIdLP$ = this.closestImageIdLPSource.asObservable();
  paintingNameLP$ = this.paintingNameLPSource.asObservable();
  fileNameLP$ = this.fileNameLPSource.asObservable();
  closestImageDataLP$ = this.closestImageDataLPSource.asObservable();
  closeImagesLP$ = this.closeImagesLPSource.asObservable();
  closenessThresholdLP$ = this.closenessThresholdLPSource.asObservable();

  setArtistName(name: string) {
    this.artistNameSource.next(name);
  }

  setSimilarity(similarity: string) {
    this.similaritySource.next(similarity);
  }

  setClosestImageId(id: string) {
    this.closestImageIdSource.next(id);
  }

  setPaintingName(name: string) {
    this.paintingNameSource.next(name);
  }

  setFileName(name: string) {
    this.fileNameSource.next(name);
  }

  setClosestImageData(imageData: string) {
    this.closestImageDataSource.next(imageData);
  }

  setArtistImages(images: any[]) {
    this.artistImagesSource.next(images);
  }

  setCloseImages(images: any[]) {
    this.closeImagesSource.next(images);
  }

  setClosenessThresold(threshold: string) {
    this.closenessThresoldSource.next(threshold);
  }

  setAverageDistance(distance: string) {
    this.averageDistanceSource.next(distance);
  }

  // Lost Paintings Setters:

  setArtistNameLP(name: string) {
    this.artistNameLPSource.next(name);
  }

  setSimilarityLP(similarity: string) {
    this.similarityLPSource.next(similarity);
  }

  setClosestImageIdLP(id: string) {
    this.closestImageIdLPSource.next(id);
  }

  setPaintingNameLP(name: string) {
    this.paintingNameLPSource.next(name);
  }

  setFileNameLP(name: string) {
    this.fileNameLPSource.next(name);
  }
  
  setClosestImageDataLP(data: string) {
    this.closestImageDataLPSource.next(data);
  }

  setCloseImagesLP(images: any[]) {
    this.closeImagesLPSource.next(images);
  }

  setClosenessThresholdLP(threshold: string) {
    this.closenessThresholdLPSource.next(threshold);
  }

}
