#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
vector<int> h, e, l, o;
int cnt[5];
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == 'h')
      h.push_back(i);
    if (s[i] == 'e')
      e.push_back(i);
    if (s[i] == 'l')
      l.push_back(i);
    if (s[i] == 'o')
      o.push_back(i);
  }
  if (h.size() != 0 && o.size() != 0) {
    cnt[0] = h[0];
    cnt[4] = o[o.size() - 1];
  }
  for (int i = 0; i < e.size(); ++i)
    if (e[i] > cnt[0]) {
      cnt[1] = e[i];
      break;
    }
  for (int i = l.size() - 1; i >= 0; --i)
    if (l[i] < cnt[4]) {
      cnt[3] = l[i];
      break;
    }
  for (int i = 0; i < l.size(); ++i)
    if (l[i] < cnt[3] && l[i] > cnt[1]) {
      cnt[2] = l[i];
      break;
    }
  if (cnt[0] < cnt[1] && cnt[1] < cnt[2] && cnt[2] < cnt[3] &&
      cnt[3] < cnt[4]) {
    cout << "YES";
    return 0;
  }
  cout << "NO";
  return 0;
}
