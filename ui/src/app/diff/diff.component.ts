import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


export interface Tile {
  color: string[];
  cols: number;
  rows: number;
  text?: string[] | undefined;
}

@Component({
  selector: 'app-diff',
  templateUrl: './diff.component.html',
  styleUrls: ['./diff.component.css']
})
export class DiffComponent {
  cpp = [`#include <iostream>

using namespace std;

int main() {
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  cout << "This is project Martial!";
  // But what about longer comments, splitted?
  return 0;
}
`, `#include <iostream>

using namespace std;

int main() {
  // With a comment.
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  cout << "This is project Martial!";
  /* But
  what
  about */
  cout << "Still going on..."
  // longer
  // comments
  // splitted?
  return 0;
}
`]

  file_upload_tiles: Tile[] = [
    { cols: 1, rows: 1, color: ['#ffe6e6'] },
    { cols: 1, rows: 1, color: ['#dcf0d5'] },
  ];

  tiles: Tile[] = [
    { text: this.cpp[0].split(/\r?\n/), cols: 1, rows: 24, color: Array.from({ length: 24 }, (_, i) => "#FDFDFD") },
    { text: this.cpp[1].split(/\r?\n/), cols: 1, rows: 24, color: Array.from({ length: 24 }, (_, i) => "#FDFDFD") },
  ];


  fileName: string[] = ["", ""]
  constructor(private http: HttpClient) { }

  onFileSelected(event, id: number) {
    const file: File = event.target.files[0];
    if (file) {
      this.fileName[id] = file.name;
      let reader = new FileReader();
      reader.onload = (evt) => {
        let data = evt.target?.result?.toString()
        if (data != undefined) {
          this.cpp[id] = data
          this.tiles[id].text = this.cpp[id].split(/\r?\n/)
          let len = this.tiles[id].text?.length
          if (len == undefined) {
            len = 1
          }
          this.tiles[id].color = Array.from({ length: len }, (_, i) => "#FDFDFD")
        }
      }
      reader.readAsBinaryString(file);
    }
  }


  onFileSelected0(event) {
    this.onFileSelected(event, 0)
  }

  onFileSelected1(event) {
    this.onFileSelected(event, 1)
  }

  render_index = 0
  MaxRowsToRender(): number {
    let l1 = 0
    let l2 = 0
    if (this.tiles[0].text != undefined) {
      l1 = this.tiles[0].text.length
    }
    if (this.tiles[1].text != undefined) {
      l2 = this.tiles[1].text.length
    }
    return Math.max(l1, l2)
  }

  createRange(number) {
    // return new Array(number);
    return new Array(number).fill(0)
      .map((n, index) => index + 1);
  }


  commentAnalysis() {
    let fileUpload = this.uploadFilesToBacked()
    fileUpload?.subscribe(resp => {
      console.log(resp);
      let get$ = this.getComments()
      get$?.subscribe(resp => {
        console.log(resp);
        for (let match in resp["comment_fuzzy_lines_files"]) {
          for (let idx in resp["comment_fuzzy_lines_files"][match]["file1"]) {
            let line = resp["comment_fuzzy_lines_files"][match]["file1"][idx] - 1
            this.tiles[0].color[line] = "#FFFF00"
          }
          for (let idx in resp["comment_fuzzy_lines_files"][match]["file2"]) {
            let line = resp["comment_fuzzy_lines_files"][match]["file2"][idx] - 1
            this.tiles[1].color[line] = "#FFFF00"
          }
        }

        for (let match in resp["comment_exact_lines_files"]) {
          for (let idx in resp["comment_exact_lines_files"][match]["file1"]) {
            let line = resp["comment_exact_lines_files"][match]["file1"][idx] - 1
            this.tiles[0].color[line] = "#FF0000"
          }
          for (let idx in resp["comment_exact_lines_files"][match]["file2"]) {
            let line = resp["comment_exact_lines_files"][match]["file2"][idx] - 1
            this.tiles[1].color[line] = "#FF0000"
          }
        }
      }
      
      
      );
    }
    );
  }

  uploadFilesToBacked(): Observable<Object> | undefined {
    console.log("uploadFilesToBacked started")
    if (this.tiles[0].text == undefined) {
      console.log("this.tiles[0].text is undefined")
      return
    }
    if (this.tiles[1].text == undefined) {
      console.log("this.tiles[1].text is undefined")
      return
    }

    const upload$ = this.http.post("http://127.0.0.1:5000/api/upload",
      { "file1": this.cpp[0], "file2": this.cpp[1] });

    console.log("uploadFilesToBacked finished")
    return upload$
  }

  getComments(): Observable<Object> | undefined {
    const get$ = this.http.get("http://127.0.0.1:5000/api/comments")
    return get$
  }
}
