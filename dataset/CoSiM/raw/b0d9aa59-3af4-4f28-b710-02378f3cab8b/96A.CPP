#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  int maxm = 0, count = 1;
  for (int i = 0; i < s.size() - 1; i++) {
    if (s.at(i) == s.at(i + 1))
      ++count;
    else {
      maxm = max(count, maxm);
      count = 1;
    }
  }
  maxm = max(count, maxm);

  // cout << max;
  if (maxm >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
