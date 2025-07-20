#include <iostream>

using namespace std;

int main() {

  int n, k, pivot, count;
  cin >> n >> k;

  int score[n];
  for (int i = 0; i < n; i++) {
    cin >> score[i];
  }

  pivot = score[k - 1];
  count = 0;

  for (int i = 0; i < n; i++) {

    if (score[i] >= pivot && score[i] > 0) {
      count++;
    }
  }

  cout << count;
  return 0;
}