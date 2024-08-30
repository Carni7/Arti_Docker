import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, catchError, throwError } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ImageService {
  private baseUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  inferArtistName(image: File): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('image', image);
    return this.http.post<any>(`${this.baseUrl}/inferImageCNN`, formData)
    .pipe(
      catchError(this.handleError)
    );
  }

  inferSimilarity(image: File, artistName: string): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('image', image);
    formData.append('artist_name', artistName);
    return this.http.post<any>(`${this.baseUrl}/inferImageSiamese`, formData).pipe(
      catchError(this.handleError)
    );
  }

  inferLostImage(image: File): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('image', image);
    return this.http.post<any>(`${this.baseUrl}/inferImageSiameseLostPaintings`, formData)
    .pipe(
      catchError(this.handleError)
    );
  }

  getImageUrl(id: string): string {
    return `${this.baseUrl}/image/${id}`;
  }

  private handleError(error: HttpErrorResponse) {
    console.error('Server Error:', error);
    return throwError(() => new Error('Something went wrong with the request.'));
  }
}
