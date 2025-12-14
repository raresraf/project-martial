#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define pb push_back
#define vc vector<char>

string newword(string s) {
  vc a(s.begin(), s.end());

  int p = a.size();
  string ans = "";
  if (p > 10) {
    // long word
    char start = a[0];
    char end = a[a.size() - 1];
    int count = 0;
    for (size_t i = 1; i < a.size() - 1; i++) {
      count += 1;
    }
    ans += start + to_string(count) + end;
  } else {
    // return the string as it is
    ans += s;
  }
  return ans;
}

int main() {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    cout << newword(s) << "\n";
  }
  return 0;
}
