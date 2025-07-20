#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  vector<char> ans;
  for (int i = 0; i < str.size(); i++)
    ans.push_back(str[i]);
  for (int i = 0; i < ans.size(); i++)
    cout << ans[i];
  cout << endl;

  for (int i = 0; i < ans.size(); i++) {
    if ((ans[i] == 'a' || ans[i] == 'e' || ans[i] == 'i' || ans[i] == 'o' ||
         ans[i] == 'u') ||
        (ans[i] == 'A' || ans[i] == 'B' || ans[i] == 'I' || ans[i] == 'O' ||
         ans[i] == 'U')) {
      ans.pop_back();
    }

    else if ((ans[i] > 'a' && ans[i] < 'z' &&
              !(ans[i] == 'a' || ans[i] == 'e' || ans[i] == 'i' ||
                ans[i] == 'o' || ans[i] == 'u')) ||
             ((ans[i] > 'A' && ans[i] < 'Z') &&
              !(ans[i] == 'A' || ans[i] == 'E' || ans[i] == 'I' ||
                ans[i] == 'O' || ans[i] == 'U'))) {
      ans.push_back('.');
      if (ans[i] > 'a' && ans[i] < 'z') {
      } else
        ans.push_back(ans[i] - 32);
    }
  }
  for (int i = 0; i < ans.size(); i++)
    cout << ans[i];
  return 0;
}
