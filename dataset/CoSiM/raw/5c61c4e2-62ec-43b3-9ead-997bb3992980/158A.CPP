#include <iostream>

using namespace std;

int main(int argc, const char *argv[]) {
  int n, k, count;
  cin >> n;
  cin >> k;

  int scores[n];
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  count = 0;
  for (int i = 0; i < n; i++) {
    if (scores[i] >= scores[k - 1] && scores[i] != 0)
      count++;
  }
  cout << count;

  return 0;
}
