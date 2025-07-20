#include <iostream>
#include <regex>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  regex r("[1]{7}");
  string a;
  cin >> a;
  if (regex_match(a, r)) {
    cout << "yo" << endl;
  }

  return 0;
}
