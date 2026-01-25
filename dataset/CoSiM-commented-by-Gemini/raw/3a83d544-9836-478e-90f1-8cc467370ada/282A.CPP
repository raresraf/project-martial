#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    if (s.at(1) == '+')
      ans++;
    else
      ans--;
  }
  printf("%d\n", ans);
  return 0;
}
