#include <iostream>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k; // n participants, k-th place cutoff

  int scores[n];
  int flag = n + 1; // any value greater than n

  // input
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  // checking for first non positive number
  for (int i = 0; i < n; i++) {
    if (scores[i] <= 0) {
      flag = i;
      break;
    }
  }

  if (k < flag) {
    int i = 0;
    do {
      if (scores[k + i - 1] == scores[k - 1])
        i++;
      else
        break;
    } while (i <= (n - k));
    cout << k + i - 1;
  } else {
    cout << flag;
  }
  return 0;
}
