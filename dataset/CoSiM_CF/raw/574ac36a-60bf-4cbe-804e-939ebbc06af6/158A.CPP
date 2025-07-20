#include <iostream>
using namespace std;
int main() {
  int result, i, k, n, score[50];
  cin >> n >> k;

  for (i = 0; i < n; i++) {
    cin >> score[i];
  }
  for (i = 0; i < n; i++) {
    if (score[i] == 0) {
      result = i;
      break;
    }
    if (i >= k - 1) {
      if (score[i] != score[i + 1]) {
        result = i + 1;
        break;
      }
    }
  }
  cout << result;
  return 0;
}
