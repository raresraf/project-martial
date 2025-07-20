#include <iostream>
#include <string>

using namespace std;
bool is_hello(const string &);

int main(void) {
  string input;
  cin >> input;

  if (is_hello(input))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

bool is_hello(const string &str) {

  bool found_o = false, found_l1 = false, found_l2 = false, found_e = false,
       found_h = false;

  for (int i = str.length() - 1; i >= 0; i--)
    if (!found_o && str.at(i) == 'o')
      found_o = true;

    else if (found_o && !found_l2 && str.at(i) == 'l')
      found_l2 = true;

    else if (found_o && found_l2 && !found_l1 && str.at(i) == 'l')
      found_l1 = true;

    else if (found_o && found_l2 && found_l1 && !found_e && str.at(i) == 'e')
      found_e = true;

    else if (found_o && found_l2 && found_l1 && found_e && !found_h &&
             str.at(i) == 'h')
      found_h = true;

  if (found_o && found_l1 && found_l2 && found_e && found_h)
    return true;
  else
    return false;
}