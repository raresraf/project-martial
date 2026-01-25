#include <iostream>
using namespace std;

int main() {
  int n, k;
  int i = 0;
  int scores[50] = {0};
  cin >> n >> k;

  for (i = 0; i < n; i++) {
    cin >> scores[i];
  }

  i = k - 1;
  if (scores[i] == 0) {
    while (i >= 0 && scores[i] == 0) {
      i--;
    }
  }

  else {
    while (i < n - 1 && scores[i + 1] == scores[i]) {
      i++;
    }
  }
  cout << i + 1;
}
