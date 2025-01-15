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
  string x;
  cin >> x;
  bool h, e, l1, l2, o;
  h = e = l1 = l2 = o = false;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] == 'h')
      h = true;
    if (x[i] == 'e' && h)
      e = true;
    if (x[i] == 'l' && h && e && l1)
      l2 = true;
    if (x[i] == 'l' && h && e)
      l1 = true;
    if (x[i] == 'o' && h && e && l1 && l2)
      o = true;
  }
  // printf("%d %d %d %d %d\n",h,e,l1,l2,o);
  if (h && e && l1 && l2 && o)
    puts("YES");
  else
    puts("NO");
  return 0;
}