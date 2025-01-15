/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h>

using namespace std;

#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

#define D(x) cout << #x << " = " << x << endl;

int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  string x, y;
  cin >> x >> y;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] >= 'a' && x[i] <= 'z')
      x[i] = x[i] - 'a' + 'A';
    if (y[i] >= 'a' && y[i] <= 'z')
      y[i] = y[i] - 'a' + 'A';
  }
  // cout << x << " " << y << endl;
  if (x == y)
    printf("0\n");
  else {
    if (x > y)
      printf("1\n");
    else
      printf("-1\n");
  }
  return 0;
}