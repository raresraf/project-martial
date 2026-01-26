#include <iostream>
#include <vector>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  int N, orig;
  cin >> N;
  vector<int> taken;
  orig = N;
  for (int i = 1; i <= orig; i++) {
    if (N < i) {
      taken[taken.size() - 1] += N;
      break;
    }
    N -= i;
    taken.push_back(i);
  }
  cout << (int)taken.size() << "\n";
  cout << taken[0];
  for (int i = 1; i < (int)taken.size(); i++)
    cout << " " << taken[i];
  cout << "\n";
  return 0;
}
