#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  char ch[3100], ip = 0;
  int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0,
      k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0,
      u = 0, v = 0, w = 0, x = 0, y = 0, z = 0;
  int lp, cnt = 0;

  cin >> ch;

  for (lp = 0; lp < strlen(ch); lp++) {
    if (ch[lp] >= 'a' && ch[lp] <= 'z') {
      if (ch[lp] == 'a' && a != 6) {
        cnt++;
        a = 6;
      } else if (ch[lp] == 'b' && b != 6) {
        cnt++;
        b = 6;
      } else if (ch[lp] == 'c' && c != 6) {
        cnt++;
        c = 6;
      } else if (ch[lp] == 'd' && d != 6) {
        cnt++;
        d = 6;
      } else if (ch[lp] == 'e' && e != 6) {
        cnt++;
        e = 6;
      } else if (ch[lp] == 'f' && f != 6) {
        cnt++;
        f = 6;
      } else if (ch[lp] == 'g' && g != 6) {
        cnt++;
        g = 6;
      }

      else if (ch[lp] == 'h' && h != 6) {
        cnt++;
        h = 6;
      }

      else if (ch[lp] == 'i' && i != 6) {
        cnt++;
        i = 6;
      }

      else if (ch[lp] == 'j' && j != 6) {
        cnt++;
        j = 6;
      }

      else if (ch[lp] == 'k' && k != 6) {
        cnt++;
        k = 6;
      }

      else if (ch[lp] == 'l' && l != 6) {
        cnt++;
        l = 6;
      }

      else if (ch[lp] == 'm' && m != 6) {
        cnt++;
        m = 6;
      }

      else if (ch[lp] == 'n' && n != 6) {
        cnt++;
        n = 6;
      }

      else if (ch[lp] == 'o' && o != 6) {
        cnt++;
        o = 6;
      }

      else if (ch[lp] == 'p' && p != 6) {
        cnt++;
        p = 6;
      }

      else if (ch[lp] == 'q' && q != 6) {
        cnt++;
        q = 6;
      }

      else if (ch[lp] == 'r' && r != 6) {
        cnt++;
        r = 6;
      }

      else if (ch[lp] == 's' && s != 6) {
        cnt++;
        s = 6;
      }

      else if (ch[lp] == 't' && t != 6) {
        cnt++;
        t = 6;
      }

      else if (ch[lp] == 'u' && u != 6) {
        cnt++;
        u = 6;
      }

      else if (ch[lp] == 'v' && v != 6) {
        cnt++;
        v = 6;
      }

      else if (ch[lp] == 'w' && w != 6) {
        cnt++;
        w = 6;
      }

      else if (ch[lp] == 'x' && x != 6) {
        cnt++;
        x = 6;
      }

      else if (ch[lp] == 'y' && y != 6) {
        cnt++;
        y = 6;
      }

      else if (ch[lp] == 'z' && z != 6) {
        cnt++;
        z = 6;
      }
    }
  }

  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}