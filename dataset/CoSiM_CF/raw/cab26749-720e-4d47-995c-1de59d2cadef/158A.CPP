// my first program in C++
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int i = 0;
  int threshold = -1;
  int currentNr = 0;
  for (size_t i = 0; i < k; i++) {
    cin >> threshold;
    if (threshold <= 0) {
      break;
    }
    currentNr++;
  }
  // cout << currentNr << endl;

  int nr;
  cin >> nr;
  while (nr > 0 && currentNr < n - 1 && threshold == nr) {
    currentNr++;
    cin >> nr;
  }
  if (nr > 0 && threshold == nr)
    currentNr++;

  // cout << threshold<<endl;
  // if (threshold == -1) {
  // 	cout << n;
  // } else {
  // 	cout << currentNr;
  // }
  cout << currentNr;
  return 0;
}