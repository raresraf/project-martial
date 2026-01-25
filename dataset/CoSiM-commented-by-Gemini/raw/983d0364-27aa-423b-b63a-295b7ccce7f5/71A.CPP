#include <iostream>
#include <string>
using namespace std;
using std::getline;
using std::string;

int main() {
  int x;
  cin >> x;
  for (int i = 0; i < x; i++) {
    string n;
    cin >> n;
    int tam = n.length();
    if (tam <= 10) {
      cout << n << endl;
    } else {
      cout << n.at(0) << tam - 2 << n.at(tam - 1) << endl;
    }
  }
}