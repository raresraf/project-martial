#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;

int main(void) {
  int n, k;
  vector<int> scores;

  cin >> n >> k;
  int count = 0;
  int score;

  while (count++ < n) {
    cin >> score;
    scores.push_back(score);
  }
  count = 0;
  int index;
  for (index = 0; index < n && index < k; count++) {
    if (scores[index++] <= 0) {
      break;
    }
  }
  while (index < n && scores[index] > 0 && scores[index - 1] == scores[index]) {
    index++;
    count++;
  }
  cout << count;

  return 0;
}