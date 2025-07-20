#include <iostream>

using namespace std;

int main() {
  int n, k, score, minToAdvance;
  int promotedContenstants = 0;

  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> score;
    if (score && i < k - 1)
      promotedContenstants++;
    else if (score && i == k - 1) {
      promotedContenstants++;
      minToAdvance = score;
    } else if (score && i > k - 1)
      if (score == minToAdvance)
        promotedContenstants++;
  }

  cout << promotedContenstants << endl;

  return 0;
}
