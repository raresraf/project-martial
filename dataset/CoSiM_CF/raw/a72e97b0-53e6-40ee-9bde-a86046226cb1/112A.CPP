#include <bits/stdc++.h>
#include <string.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  char s1[MAX], s2[MAX];
  int i;
  cin >> s1 >> s2;
  for (i = 0; i < strlen(s1); i++) {
    if (tolower(s1[i]) > tolower(s2[i])) {
      cout << 1;
      return 0;
    }
    if (tolower(s1[i]) < tolower(s2[i])) {
      cout << -1;
      return 0;
    }
  }
  cout << 0;
  return 0;
}
