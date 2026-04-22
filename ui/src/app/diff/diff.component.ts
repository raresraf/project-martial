import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ProgressBarMode } from '@angular/material/progress-bar';
import {
  LLM_DEFAULT_PROMPT, LLM_DEMO_LEFT, LLM_DEMO_RIGHT,
  LLM_ANNOTATED_LEFT, LLM_ANNOTATED_RIGHT, computeAnnotationColors,
} from './llm-data';

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
      this.similarity = NaN;
      this.llmIsAnnotated = false;
      this.llmAnnotatedWithModel = '';
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

  resetTileColors() {
    for (let i = 0; i < this.tiles.length; i++) {
      this.tiles[i].color = this.tiles[i].color.map(() => "#FDFDFD")
    }
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
        this.resetTileColors()
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
        this.resetTileColors()
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


  networkTrafficAnalysis() {
    this.progress_bar_mode = "indeterminate"
    let fileUpload = this.uploadFilesToBackend()
    fileUpload?.subscribe(resp => {
      console.log(resp);
      let get$ = this.runNetworkTrafficAnalysis(this.selectedNgram)
      get$?.subscribe(resp => {
        console.log(resp);
        this.resetTileColors()
        this.colorBasedOnResp(resp, "identical", ["#EE82EE", "#EE83EE", "#EE84EE", "#EE85EE"])
        this.colorBasedOnResp(resp, "complexity", ["#FF0000", "#EF0000", "#DF0000", "#CF0000"])
        this.similarity = resp["similarity"]
        this.backendSelectedNgram = resp["selected_ngram"]
        this.progress_bar_mode = "determinate"
      });
    });
  }

  runNetworkTrafficAnalysis(ngram: number): Observable<Object> | undefined {
    const get$ = this.http.get(`http://127.0.0.1:5000/api/network_traffic_analysis?ngram=${ngram}`)
    return get$
  }


  chosenAnalysis: string = "";
  selectedNgram: number = 4;
  SomethingEnabled() {
    return this.chosenAnalysis != ""
  }
  selectedCommentExample = 'hello-world';

  CommentsEntry() {
    this.selectedCommentExample = 'hello-world';
    this.mockInputFilesComments()
    this.chosenAnalysis = "comments"
  }

  loadCommentExample(id: string) {
    this.selectedCommentExample = id;
    const pairs: Record<string, [string, string]> = {
      'hello-world': [
        `#include <iostream>

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
`,
        `#include <iostream>
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
`,
      ],
      'go-kubernetes-96': [
        `// Package validation_test tests Kubernetes object validation logic.
// These tests ensure that invalid configurations are properly rejected.
package validation_test

import "testing"

// TestValidateServicePort checks service port validation rules.
// Valid ports must be in range [1, 65535].
func TestValidateServicePort(t *testing.T) {
	// Test boundary conditions for port numbers
	cases := []struct {
		port    int32
		isValid bool
	}{
		{80, true},     // Standard HTTP port - valid
		{0, false},     // Port 0 is reserved - invalid
		{65535, true},  // Maximum valid port
		{65536, false}, // Exceeds maximum - invalid
	}
	for _, c := range cases {
		err := validatePort(c.port)
		if (err == nil) != c.isValid {
			t.Errorf("port %d: expected valid=%v", c.port, c.isValid)
		}
	}
}

// TestValidateLabel verifies that label keys follow Kubernetes naming rules.
// Labels must consist of alphanumeric characters, dashes, underscores, or dots.
func TestValidateLabel(t *testing.T) {
	// Valid labels should pass without errors
	validLabels := []string{"app", "my-app", "v1.2", "env_prod"}
	for _, l := range validLabels {
		if errs := validateLabelKey(l); len(errs) != 0 {
			t.Errorf("expected %q to be valid, got %v", l, errs)
		}
	}
	// Labels with illegal characters should fail validation
	if errs := validateLabelKey("invalid/key"); len(errs) == 0 {
		t.Error("expected invalid/key to fail validation")
	}
}
`,
        `// Package validation_test contains tests for Kubernetes API validation.
// Covers edge cases and boundary conditions for field validators.
package validation_test

import "testing"

// TestValidateContainerPort verifies container port validation.
// Container ports follow the same rules: range [1, 65535].
func TestValidateContainerPort(t *testing.T) {
	// Define test scenarios covering valid and invalid ports
	scenarios := []struct {
		portNum int32
		valid   bool
	}{
		{8080, true},    // Application port - valid
		{-1, false},     // Negative port - invalid
		{443, true},     // HTTPS port - valid
		{99999, false},  // Port too large - invalid
	}
	for _, s := range scenarios {
		result := validateContainerPort(s.portNum)
		if (result == nil) != s.valid {
			t.Errorf("port %d: wanted valid=%v", s.portNum, s.valid)
		}
	}
}

// TestValidateAnnotation checks annotation key and value constraints.
// Annotation keys must be valid DNS subdomain names or simple identifiers.
func TestValidateAnnotation(t *testing.T) {
	// Annotation values can be any string, including empty
	validAnnotations := map[string]string{
		"description":              "a useful service",
		"prometheus.io/scrape":     "true",
		"kubectl.kubernetes.io/last-applied-configuration": "{}",
	}
	for k, v := range validAnnotations {
		if errs := validateAnnotation(k, v); len(errs) != 0 {
			t.Errorf("annotation %q=%q: unexpected errors %v", k, v, errs)
		}
	}
}
`,
      ],
      'cpp-codeforces-1595': [
        `#include <bits/stdc++.h>
using namespace std;

// Lucky numbers contain only digits 4 and 7.
// This function checks if a given number qualifies as lucky.
bool isLucky(int n) {
    // Repeatedly extract the last digit and verify it
    while (n > 0) {
        int d = n % 10;
        // Only digits 4 and 7 are allowed in lucky numbers
        if (d != 4 && d != 7) return false;
        n /= 10;
    }
    return true;
}

// Count how many lucky numbers exist in range [1, n].
// Uses brute force since lucky numbers are sparse.
int countLucky(int n) {
    int cnt = 0;
    // Generate all lucky numbers up to n by treating 4/7 as binary digits
    for (int i = 1; i <= n; i++) {
        if (isLucky(i)) cnt++;
    }
    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Read query count
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        // Output count of lucky numbers up to n
        cout << countLucky(n) << "\\n";
    }
    return 0;
}
`,
        `#include <iostream>
#include <string>
using namespace std;

// A number is "lucky" if all its digits are exactly 4 or 7.
// Returns true if the number is lucky, false otherwise.
bool lucky(const string& s) {
    // Iterate through each character in the number
    for (char c : s) {
        // Any digit other than 4 or 7 makes the number non-lucky
        if (c != '4' && c != '7') return false;
    }
    return true;
}

// Generate all lucky numbers with at most maxLen digits.
// Recursively builds strings containing only 4s and 7s.
void generate(string cur, int maxLen, vector<string>& res) {
    if (!cur.empty()) res.push_back(cur);
    // Base case: reached maximum length
    if ((int)cur.size() == maxLen) return;
    // Branch on adding digit 4 or digit 7
    generate(cur + "4", maxLen, res);
    generate(cur + "7", maxLen, res);
}

int main() {
    // Pre-generate all lucky numbers up to 10 digits
    vector<string> lucky_nums;
    generate("", 10, lucky_nums);
    int t;
    cin >> t;
    while (t--) {
        string n;
        cin >> n;
        // Count generated lucky numbers that are <= n
        int cnt = 0;
        for (auto& l : lucky_nums) {
            if (l.size() < n.size() || (l.size() == n.size() && l <= n)) cnt++;
        }
        // Print the result for this query
        cout << cnt << "\\n";
    }
    return 0;
}
`,
      ],
      'java-elastic-2344': [
        `package org.elasticsearch.xpack.ql.execution.search;

/**
 * Provides physical operation implementations for Elasticsearch queries.
 * Maps logical query operations to their Elasticsearch-specific counterparts.
 */
public class EsPhysicalOperationProviders {

    /**
     * Creates a filter operation wrapping an Elasticsearch query.
     * The filter is pushed down to the shard level for efficiency.
     *
     * @param filter the logical filter expression to translate
     * @return a physical operation backed by an ES query
     */
    public PhysicalOperation filterOperation(Filter filter) {
        // Translate the logical filter into an ES query builder
        QueryBuilder qb = toQueryBuilder(filter.condition());
        // Wrap the query in an executable plan node
        return new EsQueryExec(filter.child(), qb);
    }

    /**
     * Translates a logical sort order to an Elasticsearch SortBuilder.
     * Null handling follows ANSI SQL semantics: nulls last for ASC order.
     *
     * @param order the logical ordering expression
     * @return an ES SortBuilder configured with matching sort direction
     */
    private SortBuilder<?> toSort(Order order) {
        // Map the logical direction to ES sort order enum
        SortOrder esOrder = order.direction() == ASC ? SortOrder.ASC : SortOrder.DESC;
        return SortBuilders.fieldSort(order.child().toString()).order(esOrder);
    }
}
`,
        `package org.elasticsearch.xpack.ql.util;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for query utility functions and physical operation providers.
 * Verifies correct translation of logical plans to Elasticsearch operations.
 */
public class UtilTests {

    /**
     * Tests that filterOperation correctly wraps a logical filter condition.
     * The resulting physical operation should be an EsQueryExec node.
     *
     * @throws Exception if the translation fails unexpectedly
     */
    @Test
    public void testFilterTranslation() throws Exception {
        // Build a simple equality filter for testing
        Filter filter = new Filter(source(), EMPTY, new Equals(source(), field, literal));
        // Translate to an Elasticsearch physical operation
        PhysicalOperation op = new EsPhysicalOperationProviders().filterOperation(filter);
        // Verify the output is a properly constructed ES query node
        assertNotNull("Physical operation should not be null", op);
        assertEquals("Should produce an EsQueryExec node",
            EsQueryExec.class, op.getClass());
    }

    /**
     * Tests null handling in sort translation.
     * Ascending order should produce SortOrder.ASC in the ES sort builder.
     */
    @Test
    public void testSortTranslation() {
        // Create an ascending order expression pointing to a field
        Order asc = new Order(source(), field, ASC, LAST);
        // Translate to ES sort and verify the direction is preserved
        SortBuilder<?> sort = new EsPhysicalOperationProviders().toSort(asc);
        assertEquals("Ascending order should map to SortOrder.ASC",
            SortOrder.ASC, ((FieldSortBuilder) sort).order());
    }
}
`,
      ],
    };
    const [left, right] = pairs[id] ?? pairs['hello-world'];
    this.inputFiles = [left, right];
    this.setupTiles(Math.max(left.split(/\r?\n/).length, right.split(/\r?\n/).length));
    this.similarity = NaN;
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

  NetworkTrafficAnalysisEntry() {
    this.mockInputFilesNetworkTrafficAnalysis()
    this.chosenAnalysis = "networkTrafficAnalysis"
    this.selectedNgram = 4;
  }
  EnableNetworkTrafficAnalysis() {
    return this.chosenAnalysis == "networkTrafficAnalysis"
  }

  LlmAnnotateEntry() {
    this.mockInputFilesLlmAnnotate();
    this.chosenAnalysis = "llmAnnotate";
  }
  EnableLlmAnnotate() {
    return this.chosenAnalysis === "llmAnnotate";
  }
  llmAnnotate() {
    this.progress_bar_mode = "indeterminate";
    const model = this.llmSelectedModel;
    setTimeout(() => {
      const al = LLM_ANNOTATED_LEFT[model] ?? LLM_ANNOTATED_LEFT['gemini-1.5-pro'];
      const ar = LLM_ANNOTATED_RIGHT[model] ?? LLM_ANNOTATED_RIGHT['gemini-1.5-pro'];
      const leftLines = al.split(/\r?\n/);
      const rightLines = ar.split(/\r?\n/);
      this.tiles[0].text = leftLines;
      this.tiles[0].color = computeAnnotationColors(al);
      this.tiles[1].text = rightLines;
      this.tiles[1].color = computeAnnotationColors(ar);
      this.llmIsAnnotated = true;
      this.llmAnnotatedWithModel = this.llmModels.find(m => m.id === model)?.name ?? model;
      this.progress_bar_mode = "determinate";
    }, 2200);
  }
  resetLlmAnnotation() {
    const leftLines = this.inputFiles[0].split(/\r?\n/);
    const rightLines = this.inputFiles[1].split(/\r?\n/);
    this.tiles[0].text = leftLines;
    this.tiles[0].color = leftLines.map(() => '#FDFDFD');
    this.tiles[1].text = rightLines;
    this.tiles[1].color = rightLines.map(() => '#FDFDFD');
    this.llmIsAnnotated = false;
    this.llmAnnotatedWithModel = '';
  }
  onLlmModelChange() {
    if (this.llmIsAnnotated) {
      this.resetLlmAnnotation();
    }
  }
  toggleLlmPrompt() {
    this.llmPromptExpanded = !this.llmPromptExpanded;
  }
  resetLlmPrompt() {
    this.llmSystemPrompt = LLM_DEFAULT_PROMPT;
  }

  similarity: number = NaN;
  backendSelectedNgram: number;

  llmModels = [
    { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
    { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
    { id: 'gpt-4o', name: 'GPT-4o' },
    { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet' },
    { id: 'claude-3-opus', name: 'Claude 3 Opus' },
  ];
  llmSelectedModel = 'gemini-1.5-pro';
  llmSystemPrompt = LLM_DEFAULT_PROMPT;
  llmPromptExpanded = false;
  llmIsAnnotated = false;
  llmAnnotatedWithModel = '';
  UpdatedSimilarity() {
    return !isNaN(this.similarity)
  }

  GetColorUpdatedSimilarity(): string {
    if (this.similarity >= 0.7)
      return "#0F9D58"
    if (this.similarity >= 0.4)
      return "#F4B400"
    return "#DB4437"
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
      case "networkTrafficAnalysis": {
        this.networkTrafficAnalysis();
        break;
      }
      case "llmAnnotate": {
        this.llmAnnotate();
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
    this.loadCommentExample('hello-world');
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
    this.setupTiles(Math.max(this.inputFiles[0].split(/\r?\n/).length, this.inputFiles[1].split(/\r?\n/).length))
  }

  mockInputFilesNetworkTrafficAnalysis() {
    if (this.chosenAnalysis == "networkTrafficAnalysis") {
      return
    }
    this.inputFiles = [
      `5.6.51
i&"sC1lL
v.~UlB@]1}Dy
mysql_native_password`, 
      `select @@version_comment limit 1
root
mysql_native_password
_os        macos13.6
_platform  x86_64
_client_version  8.0.39
_client_name     libmysql
_pid       40541
os_user    raresraf
program_name     mysql
5.7.44-google-log
mysql_native_password`,
    ]
    this.setupTiles(Math.max(this.inputFiles[0].split(/\r?\n/).length, this.inputFiles[1].split(/\r?\n/).length))
  }

  mockInputFilesLlmAnnotate() {
    if (this.chosenAnalysis === "llmAnnotate") {
      return;
    }
    this.inputFiles = [LLM_DEMO_LEFT, LLM_DEMO_RIGHT];
    this.setupTiles(Math.max(
      LLM_DEMO_LEFT.split(/\r?\n/).length,
      LLM_DEMO_RIGHT.split(/\r?\n/).length,
    ));
    this.llmIsAnnotated = false;
    this.llmAnnotatedWithModel = '';
  }
}

