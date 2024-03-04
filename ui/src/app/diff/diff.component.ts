import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ProgressBarMode } from '@angular/material/progress-bar';

export interface Tile {
  color: string[];
  cols: number;
  rows: number;
  text?: string[] | undefined;
}

@Component({
  selector: 'app-diff',
  templateUrl: './diff.component.html',
  styleUrls: ['./diff.component.css'],
})
export class DiffComponent {
  inputFiles = [``, ``]

  file_upload_tiles = [
    { cols: 1, rows: 1, color: ['#ffe6e6'] },
    { cols: 1, rows: 1, color: ['#dcf0d5'] },
  ];

  tiles = [
    { text: this.inputFiles[0].split(/\r?\n/), cols: 1, rows: 2, color: Array.from({ length: 2 }, (_, i) => "#FDFDFD") },
    { text: this.inputFiles[1].split(/\r?\n/), cols: 1, rows: 2, color: Array.from({ length: 2 }, (_, i) => "#FDFDFD") },
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
          this.inputFiles[id] = data
          this.tiles[id].text = this.inputFiles[id].split(/\r?\n/)
          let len = this.tiles[id].text?.length
          if (len == undefined) {
            len = 1
          }
          this.tiles[id].color = Array.from({ length: len }, (_, i) => "#FDFDFD")
        }
      }
      reader.readAsBinaryString(file);
      this.similarity = NaN
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
    return new Array(number).fill(0)
      .map((n, index) => index + 1);
  }

  colorBasedOnResp(resp, label, colors) {
    let colorPick = -1
    for (let match in resp[label]) {
      colorPick++
      let pickColor = colors[colorPick % colors.length]
      for (let idx in resp[label][match]["file1"]) {
        let line = resp[label][match]["file1"][idx] - 1
        this.tiles[0].color[line] = pickColor
      }
      for (let idx in resp[label][match]["file2"]) {
        let line = resp[label][match]["file2"][idx] - 1
        this.tiles[1].color[line] = pickColor
      }
    }
  }

  progress_bar_mode: ProgressBarMode = "determinate"

  commentAnalysis() {
    this.progress_bar_mode = "indeterminate"
    let fileUpload = this.uploadFilesToBackend()
    fileUpload?.subscribe(resp => {
      console.log(resp);
      let get$ = this.getComments()
      get$?.subscribe(resp => {
        console.log(resp);
        this.colorBasedOnResp(resp, "comment_use_lines_files", ["#EE6611", "#EE6612", "#EE6613", "#EE6614"])
        this.colorBasedOnResp(resp, "comment_roberta_lines_files", ["#EE22EE", "#EE23EE", "#EE24EE", "#EE25EE"])
        this.colorBasedOnResp(resp, "comment_elmo_lines_files", ["#EE82EE", "#EE83EE", "#EE84EE", "#EE85EE"])
        this.colorBasedOnResp(resp, "comment_spacy_core_web_lines_files", ["#FFFF00", "#FFFF01", "#FFFF02", "#FFFF03"])
        this.colorBasedOnResp(resp, "comment_fuzzy_lines_files", ["#FF6600", "#FF6601", "#FF6602", "#FF6603"])
        this.colorBasedOnResp(resp, "comment_exact_lines_files", ["#FF0000", "#FF0101", "#FF0202", "#FF0303"])
        this.progress_bar_mode = "determinate"
      });
    });
  }

  uploadFilesToBackend(): Observable<Object> | undefined {
    console.log("uploadFilesToBackend started")
    if (this.tiles[0].text == undefined) {
      console.log("this.tiles[0].text is undefined")
      return
    }
    if (this.tiles[1].text == undefined) {
      console.log("this.tiles[1].text is undefined")
      return
    }

    const upload$ = this.http.post("http://127.0.0.1:5000/api/upload",
      {
        "file1": this.inputFiles[0],
        "file2": this.inputFiles[1],
      });
    this.similarity = NaN
    console.log("uploadFilesToBackend finished")
    return upload$
  }

  getComments(): Observable<Object> | undefined {
    const get$ = this.http.get("http://127.0.0.1:5000/api/comments")
    return get$
  }


  enableWord2Vec: boolean;
  enableElmo: boolean;
  enableRoberta: boolean;
  enableUse: boolean = true;

  thresholdWord2Vec: number = 0.97;
  thresholdElmo: number = 0.99;
  thresholdRoberta: number = 0.90;
  thresholdUse: number = 0.90;

  updateFlags() {
    let flags = {
      "enable_word2vec": this.enableWord2Vec,
      "enable_elmo": this.enableElmo,
      "enable_roberta": this.enableRoberta,
      "enable_use": this.enableUse,
      "threshold_word2vec": this.thresholdWord2Vec,
      "threshold_elmo": this.thresholdElmo,
      "threshold_roberta": this.thresholdRoberta,
      "threshold_use": this.thresholdUse,
    }
    const upload = this.http.post("http://127.0.0.1:5000/api/comments/flags",
      flags,
    );


    upload?.subscribe(resp => {
      console.log("updateFlags finished with flags ", flags)
      console.log(resp);
    })
  }


  rComplexityAnalysis() {
    this.progress_bar_mode = "indeterminate"
    let fileUpload = this.uploadFilesToBackend()
    fileUpload?.subscribe(resp => {
      console.log(resp);
      let get$ = this.runRComplexity()
      get$?.subscribe(resp => {
        console.log(resp);
        this.colorBasedOnResp(resp, "identical", ["#EE82EE", "#EE83EE", "#EE84EE", "#EE85EE"])
        this.colorBasedOnResp(resp, "complexity", ["#FF0000", "#EF0000", "#DF0000", "#CF0000"])
        this.similarity = resp["similarity"]
        this.progress_bar_mode = "determinate"
      });
    });
  }

  runRComplexity(): Observable<Object> | undefined {
    const get$ = this.http.get("http://127.0.0.1:5000/api/rcomplexity")
    return get$
  }


  chosenAnalysis: string = "";
  SomethingEnabled() {
    return this.chosenAnalysis != ""
  }
  CommentsEntry() {
    this.mockInputFilesComments()
    this.chosenAnalysis = "comments"
  }
  EnableComments() {
    return this.chosenAnalysis == "comments"
  }
  RComplexityEntry() {
    this.mockInputFilesRComplexity()
    this.chosenAnalysis = "rComplexity"
  }
  EnableRComplexity() {
    return this.chosenAnalysis == "rComplexity"
  }

  similarity: number = NaN;
  UpdatedSimilarity() {
    return !isNaN(this.similarity)
  }

  GetColorUpdatedSimilarity(): string {
    if (this.similarity > 0.5)
      return "#FF0000"
    return "#0000FF"
  }

  startAnalysis() {
    console.log("startAnalysis called to start: %s", this.chosenAnalysis)
    switch (this.chosenAnalysis) {
      case "comments": {
        this.commentAnalysis();
        break;
      }
      case "rComplexity": {
        this.rComplexityAnalysis();
        break;
      }
      default: {
        console.log("unknown chosenAnalysis: %s", this.chosenAnalysis)
      }
    }
  }



  setupTiles(noRows: number) {
    this.tiles = [
      { text: this.inputFiles[0].split(/\r?\n/), cols: 1, rows: noRows, color: Array.from({ length: noRows }, (_, i) => "#FDFDFD") },
      { text: this.inputFiles[1].split(/\r?\n/), cols: 1, rows: noRows, color: Array.from({ length: noRows }, (_, i) => "#FDFDFD") },
    ];
    return
  }

  mockInputFilesComments() {
    if (this.chosenAnalysis == "comments") {
      return
    }
    this.inputFiles = [`#include <iostream>

using namespace std;

int main() {
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  cout << "This is project Martial!";
  // But what about longer comments, split with small typo?
  cout << "One more log";
  // The computer was born to
  // solve problems that did not exist before.
  // Again, if it's one original sentence, yes, it's plagiarism.
  return 0;
}
`, `#include <iostream>
// The computer was created to 
// solve problems that did not exist.
using namespace std;

int main() {
  // Welcome to Project Martial!
  // 
  cout << "Hello World!";
  // If it's one original sentence, yes, it's plagiarism.
  cout << "This is project Martial!";
  /* But
  what
  about */
  cout << "Still going on..."
  // longer
  // comments
  // split with small typoss?
  cout << "One more log";
  
  // Again, if it's one original sentence, yes, it's plagiarism.
  return 0;
}
`]
    this.setupTiles(24)
  }


  mockInputFilesRComplexity() {
    if (this.chosenAnalysis == "rComplexity") {
      return
    }
    this.inputFiles = [`
{
  "metrics": {
    "branch-misses": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 12395.584888061932,
        "R-VAL": 6.770793921192811
    },
    "branches": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 359157.99653692124,
        "R-VAL": 2122.5514486821
    },
    "context-switches": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 0.0,
        "R-VAL": 0.0
    },
    "cpu-migrations": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 0.0,
        "R-VAL": 0.0
    },
    "cycles": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 2701846.99361182,
        "R-VAL": 6096.680255661326
    },
    "instructions": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 2224368.924073833,
        "R-VAL": 9532.318191246819
    },
    "page-faults": {
        "FEATURE_CONFIG": 0.1,
        "FEATURE_TYPE": "FRACTIONAL_POWER",
        "INTERCEPT": 119.63291396164364,
        "R-VAL": 3.507588986399111
    },
    "stalled-cycles-frontend": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 4133066.6385904513,
        "R-VAL": 4332.321499591121
    },
    "task-clock": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 1.7077847794125742,
        "R-VAL": 0.0026472455135430812
    }
  }
}
`, `
{
  "metrics": {
    "branch-misses": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 12109.546737935261,
        "R-VAL": 81.17416866402532
    },
    "branches": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 359560.05124197033,
        "R-VAL": 2041.1497152744814
    },
    "context-switches": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 0.0,
        "R-VAL": 0.0
    },
    "cpu-migrations": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 0.0,
        "R-VAL": 0.0
    },
    "cycles": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 2755550.209504519,
        "R-VAL": 5194.911045491396
    },
    "instructions": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 2226614.311187004,
        "R-VAL": 9270.613930687634
    },
    "page-faults": {
        "FEATURE_CONFIG": 0.4,
        "FEATURE_TYPE": "FRACTIONAL_POWER",
        "INTERCEPT": 120.62710850306229,
        "R-VAL": 3.45360717845523
    },
    "stalled-cycles-frontend": {
        "FEATURE_CONFIG": 0,
        "FEATURE_TYPE": "LOG_POLYNOMIAL",
        "INTERCEPT": 4053639.7718469757,
        "R-VAL": 63826.97522780884
    },
    "task-clock": {
        "FEATURE_CONFIG": 1,
        "FEATURE_TYPE": "POLYNOMIAL",
        "INTERCEPT": 1.7495773693153702,
        "R-VAL": 0.0020596287621877166
    }
  }
}
`]
    this.setupTiles(62)
  }
}

