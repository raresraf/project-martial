#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  cin >> a;
  vector<int> arr(4);
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == '1')
      arr[1]++;
    if (a[i] == '2')
      arr[2]++;
    if (a[i] == '3')
      arr[3]++;
  }
  vector<char> ans;
  while (arr[1]--) {
    ans.push_back('1');
    ans.push_back('+');
  }
  while (arr[2]--) {
    ans.push_back('2');
    ans.push_back('+');
  }
  while (arr[3]--) {
    ans.push_back('3');
    ans.push_back('+');
  }

  for (int i = 0; i + 1 < ans.size(); i++) {
    cout << ans[i];
  }
}