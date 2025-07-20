#include <iostream>

using namespace std;

int main() {
  int n{}, k{}, delta{}, score_k{}, next_score{}, non_null{};
  cin >> n;
  cin >> k;
  delta = n - k;
  for (int i = 0; i < k; i++) {
    cin >> score_k;
    if (score_k != 0)
      non_null++;
  }

  for (int i = 0; i < delta; i++) {
    cin >> next_score;
    if (next_score == score_k)
      k++;
  }
  if (score_k == 0)
    cout << non_null;
  else
    cout << k;

  return 0;
}