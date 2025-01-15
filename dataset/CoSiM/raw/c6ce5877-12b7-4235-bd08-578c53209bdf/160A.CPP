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
#define C(x) cout << "Chegou aqui " << x << endl;
#define pn printf("\n");
#define ps printf(" ");

int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  int n;
  cin >> n;
  int v[n];
  int som = 0, res = 0, som1 = 0;
  for (int i = 0; i < n; i++) {
    cin >> v[i];
    som += v[i];
  }
  sort(v, v + n);
  for (int i = n - 1; i >= 0; i--) {
    res++;
    som1 += v[i];
    som -= v[i];
    if (som1 > som)
      break;
  }
  printf("%d\n", res);
  return 0;
}