#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  string ans;
  vector<int> arr;
  if (s.length() == 1) {
    cout << s;
  } else {
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == '+')
        continue;
      if (s[i] == '1') {
        arr.push_back(1);
      } else if (s[i] == '2') {
        arr.push_back(2);
      } else {
        arr.push_back(3);
      }
    }
    sort(arr.begin(), arr.end());
    vector<int>::iterator ele;

    for (ele = arr.begin(); ele != arr.end(); ele++) {
      cout << *ele;
      if (ele != arr.end() - 1)
        cout << "+";
    }
  }

  return 0;
}
