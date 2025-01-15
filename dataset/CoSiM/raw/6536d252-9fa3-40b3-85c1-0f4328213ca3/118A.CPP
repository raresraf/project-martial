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
  for (int i = 0; i < x.size(); i++) {
    if (x[i] >= 'A' && x[i] <= 'Z')
      x[i] = x[i] - 'A' + 'a';
    if (x[i] == 'a' || x[i] == 'e' || x[i] == 'i' || x[i] == 'o' ||
        x[i] == 'u' || x[i] == 'y')
      continue;
    printf(".%c", x[i]);
  }
  printf("\n");

  return 0;
}