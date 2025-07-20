#include <iostream>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int ans = 0, rank = 0, pre = 101;
  int score;
  while (n--) {
    cin >> score;
    if (score == 0)
      break;

    if (rank < k) {
      ++rank;
      ++ans;
    } else if (rank == k) {
      if (score == pre)
        ++ans;
      else
        break;
    }
    pre = score;
  }
  cout << ans << endl;
  return 0;
}