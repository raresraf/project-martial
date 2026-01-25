#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long
#define INF 1000000000

using namespace std;

int len, ans;
char temp[100];
string str;

int main() {
  scanf("%d", &len);
  scanf("%s", temp);
  str = temp;

  ans = 0;
  for (int i = 1; i + 1 < str.size(); i++)
    if (str[i] == str[i - 1] || str[i] == str[i + 1]) {
      ans++;
      str = str.substr(0, i) + str.substr(i + 1);
      i--;
    }

  if (str.size() == 2 && str[0] == str[1])
    ans++;

  printf("%d\n", ans);
  return 0;
}
