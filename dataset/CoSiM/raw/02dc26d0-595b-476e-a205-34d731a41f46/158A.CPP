#include <iostream>
using namespace std;

int main() {
  int n, k;
  int score[200];
  int cnt = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++)
    cin >> score[i];
  for (int i = 0; i < n; i++) {
    if (score[i] >= score[k - 1] && score[i] > 0)
      cnt++;
    else
      break;
  }
  cout << cnt << endl;
  return 0;
}
