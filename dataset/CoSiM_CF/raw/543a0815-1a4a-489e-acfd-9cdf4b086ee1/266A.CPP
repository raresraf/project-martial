#include <bits/stdc++.h>
#include <cctype>
#include <cmath>
#include <string>
#define ll long long
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, count = 0;
  cin >> n;
  char arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  for (int i = 0; i < n; i++) {
    if (arr[i] == arr[i + 1]) {
      count++;
    }
  }
  cout << count;
}
