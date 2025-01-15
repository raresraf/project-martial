#include <algorithm>
#include <iostream>

using namespace std;

bool desc(int i, int j) { return (i > j); }

int main() {
  int n, k;
  int score[50];
  cin >> n >> k;
  k--;
  for (int i = 0; i < n; ++i)
    cin >> score[i];
  sort(score, score + n, desc);
  if (score[k] > 0) {
    while (score[k + 1] == score[k])
      k++;
  } else {
    while (k != -1 && score[k] <= 0)
      k--;
  }
  cout << k + 1;
  return 0;
}