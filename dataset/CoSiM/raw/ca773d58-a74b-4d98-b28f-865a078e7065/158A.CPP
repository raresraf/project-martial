#include <algorithm>
#include <iostream>
using namespace std;

int main() {

  int scores[50];
  int a, k, i;
  int results = 0;
  cin >> a >> k;

  for (i = 1; i <= a; i++) {
    cin >> scores[i];
  }
  int temp1 = scores[k];
  sort(scores + 1, scores + a + 1, greater<>());

  for (i = 1; i <= a; i++) {
    if (scores[i] >= temp1 && scores[i] != 0) {
      results++;
    } else {
      break;
    }
  }

  cout << results;
}
