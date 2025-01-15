//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int score;
  int kthVal = 0;
  int res = 0;
  for (int i = 0; i < n; i++) {
    cin >> score;
    if (score > 0 && res < k) {
      res++;
      if (res == k)
        kthVal = score;
    } else if (score != 0 && score == kthVal) {
      res++;
    }
  }
  cout << res;

  return 0;
}