#include <bits/stdc++.h>
#include <iostream>
#define ll long long
#define MOD 1000000007
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  int n;
  cin >> n;
  string s;
  cin >> s;
  stack<char> st;
  int cnt = 0;
  for (char ch : s) {
    if (st.empty() || st.top() != ch)
      st.push(ch);
    else
      cnt++;
  }

  cout << cnt << endl;
  return 0;
}