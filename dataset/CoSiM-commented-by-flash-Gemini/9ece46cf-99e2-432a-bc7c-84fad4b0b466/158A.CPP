#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> scores;
  int count = 0;

  while (n--) {
    int a;
    cin >> a;
    scores.emplace_back(a);
  }

  int kScore = scores[k - 1];
  sort(scores.begin(), scores.end());

  for (vector<int>::iterator it = scores.begin(); it != scores.end(); ++it) {
    int score = (*it);
    if (score >= kScore && score > 0)
      ++count;
  }
  cout << count;
  return 0;
}
