#include <iostream>
using namespace std;

int main() {
  int n, k, x;
  cin >> n;
  cin >> k;
  int students[n];
  for (int i = 0; i < n; i++) {
    cin >> x;
    students[i] = x;
  }
  int pass = 0;
  int media = students[k - 1];
  bool continua = true;
  for (int j = 0; j < n; j++ && continua) {
    if (students[j] < media) {
      continua = false;
    }
    if ((students[j] >= media) && (students[j] > 0)) {
      pass++;
    }
  }

  cout << pass;
}